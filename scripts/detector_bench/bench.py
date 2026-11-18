"""Run one candidate detector over the benchmark photos and save its raw boxes (read-only on the database).

Photos: every second admin-cabin photo plus every hand-labelled photo. Then run score.py to compare candidates.

usage:  .venv/bin/python scripts/detector_bench/bench.py NAME      -> data/exports/detector_bench/NAME.json
NAME:   <weights>@<input size>[+tta][+clahe]   for YOLO weights in models/, e.g. yolo26x@1280, yolo12x@1920+tta
        rtdetr-x@1280 | yoloe-26x-seg@1280[+p2] (text prompts) | gdino | owlv2 (Hugging Face models)"""
import json, sqlite3, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "data" / "exports" / "detector_bench"; OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = ["chair", "person", "monitor", "laptop", "backpack", "handbag", "keyboard", "mouse", "bottle", "cup", "book", "cell phone"]
PARK_PROMPTS = ["person", "car", "motorcycle", "bicycle", "truck", "bus"]
PROMPTS2 = ["office chair", "person", "computer monitor", "laptop computer", "backpack", "handbag", "computer keyboard", "computer mouse", "bottle", "cup", "book", "mobile phone", "desktop computer tower"]
PARK_PROMPTS2 = ["person", "car", "motorcycle", "scooter", "bicycle", "truck", "bus"]

def frames():
    c = sqlite3.connect(f"file:{ROOT/'data/sbi.db'}?mode=ro", uri=True)
    cab = c.execute("select frame_id, image_path, brightness, ts_local from frames where space_id='admin_cabin' and capture_status='ok' order by ts_utc").fetchall()
    lab = c.execute("select distinct f.frame_id, f.image_path, f.brightness, f.ts_local, f.space_id from labels l join frames f using(frame_id)").fetchall()
    pick = {r[0]: (r[1], r[2], r[3], "admin_cabin") for r in cab[::2]}
    pick.update({r[0]: (r[1], r[2], r[3], r[4]) for r in lab})
    return pick

def clahe(img, brightness):
    if brightness is None or brightness >= 100: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def ultra(model, img, imgsz, augment=False, conf=0.05):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5, verbose=False, device=0, augment=augment)[0]
    names = r.names
    return [(names[int(k)], round(float(c), 3), [round(float(v), 4) for v in b]) for b, c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

def make(name):
    import torch
    if name.startswith(("yolo12x", "yolo26x", "yolo11x")):
        from ultralytics import YOLO
        w, _, opt = name.partition("@"); size, *flags = opt.split("+")
        path = ROOT / "models" / f"{w}.pt"
        m = YOLO(str(path))
        if "tile" in flags:        # the whole photo plus overlapping left/right halves, each at full input size
            def tiled(img, br, space):
                h, w = img.shape[:2]; out = ultra(m, img, int(size))
                for x0, x1 in ((0, int(w * 0.6)), (int(w * 0.4), w)):
                    for name, conf, (a, b, c, d) in ultra(m, img[:, x0:x1], int(size)):
                        tw = x1 - x0
                        if (x0 > 0 and a * tw < 8) or (x1 < w and c * tw > tw - 8):      # cut by the tile edge: the other view has it whole
                            continue
                        out.append((name, conf, [round((x0 + a * tw) / w, 4), b, round((x0 + c * tw) / w, 4), d]))
                return out
            return tiled
        return lambda img, br, space: ultra(m, clahe(img, br) if "clahe" in flags else img, int(size), augment="tta" in flags)
    if name.startswith("rtdetr"):
        from ultralytics import RTDETR
        w, _, size = name.partition("@"); m = RTDETR(str(ROOT / "models" / f"{w}.pt"))
        return lambda img, br, space: ultra(m, img, int(size))
    if name.startswith("yoloe"):
        from ultralytics import YOLOE
        import os; os.chdir(ROOT / "models")
        w, _, opt = name.partition("@"); size, *flags = opt.split("+"); ms = {}
        cab = PROMPTS2 if "p2" in flags else PROMPTS
        for space, prompts in (("admin_cabin", cab), ("parking", PARK_PROMPTS2 if "p2" in flags else PARK_PROMPTS)):
            m = YOLOE(str(ROOT / "models" / f"{w}.pt")); m.set_classes(prompts, m.get_text_pe(prompts)); ms[space] = m
        return lambda img, br, space: ultra(ms[space], img, int(size))
    if name.startswith(("gdino", "owlv2")):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
        from PIL import Image
        if name.startswith("gdino"):
            repo = "IDEA-Research/grounding-dino-base"; proc = AutoProcessor.from_pretrained(repo); model = AutoModelForZeroShotObjectDetection.from_pretrained(repo).cuda().eval()
            def run(img, br, space):
                prompts = PROMPTS if space == "admin_cabin" else PARK_PROMPTS
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); h, w = img.shape[:2]
                inp = proc(images=pil, text=" . ".join(prompts) + " .", return_tensors="pt").to("cuda")
                with torch.no_grad(): o = model(**inp)
                r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[(h, w)])[0]
                labs = r.get("text_labels", r.get("labels"))
                return [(str(l), round(float(s), 3), [round(float(b[0])/w, 4), round(float(b[1])/h, 4), round(float(b[2])/w, 4), round(float(b[3])/h, 4)]) for l, s, b in zip(labs, r["scores"].tolist(), r["boxes"].tolist())]
            return run
        repo = "google/owlv2-base-patch16-ensemble"; proc = Owlv2Processor.from_pretrained(repo); model = Owlv2ForObjectDetection.from_pretrained(repo).cuda().eval()
        def run(img, br, space):
            prompts = PROMPTS if space == "admin_cabin" else PARK_PROMPTS
            h, w = img.shape[:2]; side = max(h, w)
            sq = np.full((side, side, 3), 127, np.uint8); sq[:h, :w] = img                     # OWLv2 pads to square itself; do it explicitly
            pil = Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
            inp = proc(text=[[f"a photo of a {p}" for p in prompts]], images=pil, return_tensors="pt").to("cuda")
            with torch.no_grad(): o = model(**inp)
            r = proc.post_process_grounded_object_detection(o, threshold=0.05, target_sizes=torch.tensor([[side, side]]))[0] if hasattr(proc, "post_process_grounded_object_detection") else proc.post_process_object_detection(o, threshold=0.05, target_sizes=torch.tensor([[side, side]]))[0]
            return [(prompts[int(l)], round(float(s), 3), [round(float(b[0])/w, 4), round(float(b[1])/h, 4), round(min(float(b[2]), w)/w, 4), round(min(float(b[3]), h)/h, 4)]) for l, s, b in zip(r["labels"].tolist(), r["scores"].tolist(), r["boxes"].tolist())]
        return run
    raise SystemExit(f"unknown candidate {name}")

if __name__ == "__main__":
    name = sys.argv[1]; fr = frames(); run = make(name); res = {}; t = time.time()
    for fid, (path, br, ts, space) in fr.items():
        img = cv2.imread(str(ROOT / path))
        if img is None: continue
        res[fid] = {"ts": ts, "space": space, "brightness": br, "boxes": run(img, br, space)}
    (OUT / f"{name}.json").write_text(json.dumps(res))
    print(f"{name}: {len(res)} photos in {time.time()-t:.0f}s ({(time.time()-t)/max(len(res),1):.2f} s/photo)")

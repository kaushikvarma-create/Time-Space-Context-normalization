"""Scene reader for any OpenAI-compatible vision endpoint (developed against LM Studio + Qwen3.5).

Gotchas learned from testing:
  * Qwen3.5 thinks by default. With json_schema output the JSON then lands in `reasoning_content` and
    `content` is empty; `reasoning_effort: "none"` (settings) fixes it. We still fall back to
    `reasoning_content` so a server change cannot silently yield empty records.
  * Counts move with prompt wording, so PROMPT_VERSION is stored on every frame and bumped on any edit.
  * NEVER put a list of object names in the prompt. Tested 2026-11-13: with absent decoy names in a "reuse these
    names" list, the VLM reported all six decoys in half of the cabin photos; with the open prompt it reported none.
    The old "preferred labels" hint also produced door/whiteboard/plant that the open prompt never saw.
    Naming consistency is handled afterwards, in text, by the taxonomy + curator (see sbi.curation).
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Callable

import cv2
import numpy as np
import requests

from ..errors import VLMError
from ..settings import SceneReaderSettings
from ..spaces import SpaceConfig
from .types import Relation, SceneObject, SceneReading

log = logging.getLogger(__name__)

PROMPT_VERSION = "v3"  # v3: open vocabulary (no name list), relation triples; v2: fixed status list, rubrics
STATUS_VALUES = ["present", "in_use", "idle", "moving", "parked", "open", "closed", "on", "off", "unknown"]
POSTURES = ["seated", "standing", "walking", "other"]
PREDICATES = ["sitting_on", "standing_at", "using", "holding", "carrying", "wearing", "riding", "on", "in", "near",
              "next_to", "walking_past", "parked_at"]


def prepare_image(image: np.ndarray, mask_regions, width: int) -> str:
    """Black out burned-in overlays (normalised boxes), downscale, return a base64 JPEG."""
    img = image.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in mask_regions:
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 0), -1)
    img = cv2.resize(img, (width, int(width * h / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise VLMError("could not JPEG-encode the frame")
    return base64.b64encode(buf).decode()


def build_schema(space: SpaceConfig) -> dict:
    props = {
        "people_count": {"type": "integer"},
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "posture": {"type": "string", "enum": POSTURES}, "activity": {"type": "string"}},
            "required": ["posture", "activity"]}},
        "objects": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "count": {"type": "integer"},
            "status": {"type": "string", "enum": STATUS_VALUES}}, "required": ["label", "count", "status"]}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "subject": {"type": "string"}, "predicate": {"type": "string", "enum": PREDICATES},
            "object": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["subject", "predicate", "object", "count"]}},
        **{k: {"type": "string", "enum": v} for k, v in space.scene_fields.items()},
        "summary": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def build_prompt(space: SpaceConfig) -> str:
    return (
        f"You are reading one still from a fixed camera ({space.name}, a {space.space_type}). "
        "Describe only what is visible. Do not identify anyone and do not read licence plates. "
        "objects: list EVERY distinct kind of physical object you can see (people as 'person', furniture, devices, "
        "vehicles, bags, items on desks, fixtures), using short, generic, singular, lowercase nouns "
        "(e.g. 'chair', not 'black office chair'). One entry per kind with its count and ONE status: "
        "in_use (being actively used), idle (present but unused), moving, parked (vehicle at rest), open, closed, "
        "on/off (screens, lights), present (none of the others fit), unknown (not clearly visible). "
        "If objects are packed together or partly hidden, give your best estimate of the count. "
        "relations: visible relationships between people and objects or between objects, as subject, predicate, "
        "object and count, e.g. person sitting_on chair 3, person using laptop 1, bag on floor 1. Only clearly "
        "visible ones. Scene fields: use 'unknown' for any field you cannot clearly see; do not guess. "
        "door_state: only if a door is clearly visible. "
        "tidiness: tidy = clear floor and surfaces, messy = items scattered on the floor or piled up, ok = in between."
    )


def parse_reading(text: str, space: SpaceConfig, *, model: str, latency_s: float) -> SceneReading:
    """Validate model output. Raises ValueError on an unusable shape; coerces harmless deviations."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON is not an object")
    people_count = data["people_count"]
    if not isinstance(people_count, int) or isinstance(people_count, bool) or people_count < 0:
        raise ValueError(f"invalid people_count {people_count!r}")
    objects = []
    for o in data["objects"]:
        label, count = o["label"], o["count"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"invalid object label {label!r}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"invalid count {count!r} for {label!r}")
        status = o.get("status")
        objects.append(SceneObject(label.strip(), count, status if status in STATUS_VALUES else "unknown"))
    relations = []
    for r in data.get("relations") or ():
        subj, pred, obj, n = r.get("subject"), r.get("predicate"), r.get("object"), r.get("count")
        ok = (isinstance(subj, str) and subj.strip() and isinstance(obj, str) and obj.strip() and pred in PREDICATES
              and isinstance(n, int) and not isinstance(n, bool) and n >= 1)
        if ok:   # a malformed relation is dropped, not fatal: objects and counts are still usable
            relations.append(Relation(subj.strip(), pred, obj.strip(), n))
        else:
            log.debug("dropping malformed relation %r", r)
    scene = {}
    for name, allowed in space.scene_fields.items():
        v = data.get(name)
        scene[name] = v if v in allowed else ("unknown" if "unknown" in allowed else None)
    return SceneReading(people_count=people_count, people=tuple(data.get("people") or ()),
                        objects=tuple(objects), scene=scene, summary=str(data.get("summary", "")), raw=text,
                        model=model, prompt_version=PROMPT_VERSION, relations=tuple(relations), latency_s=latency_s)


class LmStudioSceneReader:
    kind = "lmstudio_openai"

    def __init__(self, settings: SceneReaderSettings, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = settings
        self._session = session or requests.Session()
        self._sleep = sleep

    def _payload(self, image: np.ndarray, space: SpaceConfig) -> dict:
        c = self.cfg
        payload = {
            "model": c.model, "temperature": c.temperature, "max_tokens": c.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": build_prompt(space)},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + prepare_image(image, space.mask_regions, c.image_width)}}]}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "scene", "strict": True, "schema": build_schema(space)}},
        }
        if c.reasoning_effort is not None:
            payload["reasoning_effort"] = c.reasoning_effort
        return payload

    def read(self, image: np.ndarray, space: SpaceConfig) -> SceneReading:
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        payload = self._payload(image, space)
        last: Exception | None = None
        for attempt in range(1, self.cfg.retries + 1):
            started = time.monotonic()
            try:
                resp = self._session.post(url, json=payload, timeout=self.cfg.timeout_s)
            except requests.RequestException as e:           # network / timeout: transient
                last = e
            else:
                if 400 <= resp.status_code < 500:             # our request is wrong: retrying cannot help
                    raise VLMError(f"scene reader rejected the request (HTTP {resp.status_code}): {resp.text[:200]}")
                if resp.status_code >= 500:
                    last = RuntimeError(f"HTTP {resp.status_code}")
                else:
                    try:
                        msg = resp.json()["choices"][0]["message"]
                        text = msg.get("content") or msg.get("reasoning_content") or ""
                        return parse_reading(text, space, model=self.cfg.model,
                                             latency_s=round(time.monotonic() - started, 2))
                    except (ValueError, KeyError, IndexError, TypeError) as e:   # bad/empty/misshapen answer
                        last = e
            log.warning("scene reader attempt %d/%d failed: %s", attempt, self.cfg.retries, last)
            if attempt < self.cfg.retries:
                self._sleep(2 * attempt)
        raise VLMError(f"scene reader failed after {self.cfg.retries} attempts: {last}")

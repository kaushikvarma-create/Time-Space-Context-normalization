"""The labelling app as Streamlit really runs it: every page run can be on a different thread."""
import threading
from datetime import timedelta

from .conftest import REPO, FakeDetector, FakeReader, FakeSource, synthetic_image, thursday_ist
from .test_flow import DETS, cap


def test_the_labelling_app_survives_runs_on_different_threads(root, space_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from sbi.pipeline import process_frame
    from sbi.spaces import load_space
    from sbi.store.db import connect, migrate
    from sbi.store.registry import register_space
    (root / "config" / "pipeline.yaml").write_text((REPO / "config" / "pipeline.yaml").read_text())
    c = connect(root / "data" / "sbi.db")
    migrate(c)
    sp = load_space(root / "config" / "spaces" / "parking.yaml")
    register_space(c, sp, "config/spaces/parking.yaml")
    register_space(c, load_space(root / "config" / "spaces" / "admin_cabin.yaml"), "config/spaces/admin_cabin.yaml")
    for i in range(3):
        fid = cap(c, sp, root, FakeSource([synthetic_image(i)]), thursday_ist(10) + timedelta(minutes=15 * i)).frame_id
        process_frame(c, sp, fid, FakeDetector(DETS), FakeReader(), root=root)
    c.close()
    monkeypatch.setenv("SBI_ROOT", str(root))
    errors = []

    def run_once():
        at = AppTest.from_file(str(REPO / "src" / "sbi" / "labeling" / "app.py"), default_timeout=60).run()
        errors.extend(e.value for e in at.exception)

    for _ in range(2):                         # the second run reuses the cached setup from another thread
        t = threading.Thread(target=run_once)
        t.start()
        t.join()
    assert errors == []

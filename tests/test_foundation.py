"""timeutil, grid, space/settings config validation."""
from datetime import datetime, timezone

import pytest
import yaml

from sbi.errors import ConfigError
from sbi.grid import bbox_to_cell, bbox_to_cells, point_to_cell
from sbi.settings import load_nvr_credentials, load_settings
from sbi.spaces import load_space
from sbi.timeutil import time_dims

from .conftest import REPO


# ---- time (IST)
def test_time_dims_are_ist():
    d = time_dims(datetime(2026, 11, 12, 10, 26, 40, tzinfo=timezone.utc), "Asia/Kolkata")
    assert d.ts_utc == "2026-11-12T10:26:40+00:00"
    assert d.ts_local == "2026-11-12T15:56:40+05:30"
    assert (d.dow, d.dow_name, d.hour, d.slot_15, d.part_of_day, d.is_weekend) == (3, "Thu", 15, 63, "afternoon", 0)
    assert d.tz == "IST"


def test_local_date_rolls_over_at_ist_midnight_not_utc():
    d = time_dims(datetime(2026, 11, 12, 18, 30, tzinfo=timezone.utc), "Asia/Kolkata")   # 00:00 IST on the 13th
    assert (d.date, d.hour, d.part_of_day, d.dow_name) == ("2026-11-13", 0, "night", "Fri")


def test_time_dims_rejects_naive_and_bad_zone():
    with pytest.raises(ValueError):
        time_dims(datetime(2026, 1, 1), "Asia/Kolkata")
    with pytest.raises(ConfigError):
        time_dims(datetime(2026, 1, 1, tzinfo=timezone.utc), "Mars/Olympus")


# ---- grid
def test_grid_cells_and_edges():
    assert point_to_cell(0.0, 0.0, 8, 6) == (0, 0)
    assert point_to_cell(1.0, 1.0, 8, 6) == (7, 5)            # exactly on the far edge stays in-grid
    assert point_to_cell(-0.5, 2.0, 8, 6) == (0, 5)           # clamped
    assert bbox_to_cell((0.4, 0.2, 0.5, 1.0), 8, 6) == (3, 5, "R5C3")      # bottom-centre anchor
    assert bbox_to_cell((0.4, 0.2, 0.5, 1.0), 8, 6, "center")[2] == "R3C3"
    assert len(bbox_to_cells((0.1, 0.5, 0.4, 0.9), 12, 8)) == 16


# ---- space config
def test_real_space_configs_validate():
    for n in ("admin_cabin", "parking"):
        s = load_space(REPO / "config" / "spaces" / f"{n}.yaml")
        assert s.tz == "Asia/Kolkata" and s.grid.cols > 0


def _write(tmp_path, mutate):
    cfg = yaml.safe_load((REPO / "config/spaces/parking.yaml").read_text())
    mutate(cfg)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["aliases"].update(ghost="not_a_label"), "target not in vocabulary"),
    (lambda c: c["detector_map"].update(zebra="not_a_label"), "detector_map"),
    (lambda c: c.update(label_types=["nope"]), "label_types"),
    (lambda c: c.update(mask_regions=[[0.9, 0.1, 0.2, 0.2]]), "mask region"),
    (lambda c: c.update(tz="Nowhere/Land"), "timezone"),
    (lambda c: c.update(surprise=1), "surprise"),
    (lambda c: c["vocabulary"].update({"Bad Label": {"category": "x"}}), "snake_case"),
])
def test_invalid_space_configs_are_rejected_with_a_clear_message(tmp_path, mutate, needle):
    with pytest.raises(ConfigError, match=needle):
        load_space(_write(tmp_path, mutate))


def test_missing_space_file():
    with pytest.raises(ConfigError, match="not found"):
        load_space("/nonexistent/space.yaml")


# ---- settings
def _settings_root(tmp_path, extra=""):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/pipeline.yaml").write_text(
        "timezone: Asia/Kolkata\nscene_reader:\n  base_url: http://a:1\n  model: m\n" + extra)
    return tmp_path


def test_settings_load_and_env_override(tmp_path, monkeypatch):
    root = _settings_root(tmp_path)
    monkeypatch.setenv("SBI_LMSTUDIO_URL", "http://override:9")
    s = load_settings(root)
    assert s.scene_reader.base_url == "http://override:9" and s.timezone == "Asia/Kolkata"
    assert s.db_path == root.resolve() / "data/sbi.db"


def test_settings_reject_unknown_keys_and_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="invalid settings"):
        load_settings(_settings_root(tmp_path, "typo_key: 1\n"))
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "empty")


def test_credentials_missing_and_secret_never_in_repr(monkeypatch):
    for k in ("SBI_NVR_HOST", "SBI_NVR_USER", "SBI_NVR_PASS"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ConfigError, match="SBI_NVR_HOST"):
        load_nvr_credentials()
    monkeypatch.setenv("SBI_NVR_HOST", "h"); monkeypatch.setenv("SBI_NVR_USER", "u")
    monkeypatch.setenv("SBI_NVR_PASS", "s3cr3t!")
    c = load_nvr_credentials()
    assert "s3cr3t" not in repr(c) and "s3cr3t" not in str(c)
    monkeypatch.setenv("SBI_NVR_PORT", "notaport")
    with pytest.raises(ConfigError):
        load_nvr_credentials()


def test_rtsp_url_redaction(monkeypatch):
    from sbi.ingest.sources import rtsp_url
    from sbi.settings import NvrCredentials
    from sbi.spaces import CameraRef
    c = NvrCredentials(host="10.0.0.1", user="us er", password="p@ss:word")
    full = rtsp_url(c, CameraRef(channel=5))
    assert "p%40ss%3Aword" in full and "channel=5&subtype=0" in full          # special chars are percent-encoded
    red = rtsp_url(c, CameraRef(channel=5), redact=True)
    assert "p@ss" not in red and "us er" not in red and "***" in red

"""Dedupe functions and the scene reader (parsing, retries, error classes)."""
import json

import pytest
import requests

from sbi.errors import VLMError
from sbi.perception.dedupe import dedupe_contained, dedupe_same_label_iou, iou
from sbi.perception.scene_lmstudio import (STATUS_VALUES, LmStudioSceneReader, build_prompt, build_schema,
                                           parse_reading)
from sbi.settings import SceneReaderSettings

from .conftest import det, synthetic_image


# ---- dedupe
def test_same_label_overlap_is_collapsed_but_distinct_boxes_survive():
    a, b = det("bag", (0.58, 0.09, 0.65, 0.23), 0.7), det("bag", (0.58, 0.09, 0.65, 0.23), 0.28)
    far = det("bag", (0.2, 0.5, 0.3, 0.7))
    kept = dedupe_same_label_iou([a, b, far], 0.6)
    assert len(kept) == 2 and a in kept and b not in kept


def test_different_labels_never_merge():
    assert len(dedupe_same_label_iou([det("bag", (0, 0, .5, .5)), det("chair", (0, 0, .5, .5))], 0.6)) == 2


def test_contained_person_box_dropped_adjacent_people_kept():
    big = det("person", (0.40, 0.10, 0.70, 0.60), 0.9)
    inner = det("person", (0.45, 0.20, 0.60, 0.55), 0.6)             # torso box inside
    neighbour = det("person", (0.68, 0.10, 0.95, 0.60), 0.8)         # slight overlap only
    kept = dedupe_contained([big, inner, neighbour], {"person"}, 0.8)
    assert inner not in kept and big in kept and neighbour in kept


def test_contained_dedupe_only_applies_to_chosen_labels():
    outer, inner = det("chair", (0, 0, 1, 1), 0.9), det("chair", (.2, .2, .4, .4), 0.5)
    assert len(dedupe_contained([outer, inner], {"person"}, 0.8)) == 2     # chairs legitimately nest
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)


# ---- parsing
GOOD = {"people_count": 3, "people": [{"posture": "seated", "activity": "typing"}],
        "objects": [{"label": "chair", "count": 6, "status": "idle"}, {"label": "monitor", "count": 4, "status": "weird"}],
        "lighting": "artificial", "door_state": "bogus", "tidiness": "ok", "summary": "s"}


def test_parse_valid_and_coerces_harmless_deviations(space):
    r = parse_reading(json.dumps(GOOD), space, model="m", latency_s=1.0)
    assert r.people_count == 3 and [o.label for o in r.objects] == ["chair", "monitor"]
    assert r.objects[1].status == "unknown"                                   # status outside the list
    assert r.scene == {"lighting": "artificial", "door_state": "unknown", "tidiness": "ok"}   # bad enum -> unknown


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(people_count=-1), lambda d: d.update(people_count="3"), lambda d: d.update(people_count=True),
    lambda d: d["objects"].append({"label": "", "count": 1, "status": "idle"}),
    lambda d: d["objects"].append({"label": "x", "count": -2, "status": "idle"}),
    lambda d: d.pop("objects"),
])
def test_parse_rejects_unusable_shapes(space, mutate):
    d = json.loads(json.dumps(GOOD)); mutate(d)
    with pytest.raises((ValueError, KeyError)):
        parse_reading(json.dumps(d), space, model="m", latency_s=0)


def test_parse_rejects_non_json_and_non_object(space):
    for text in ("", "not json", "[1,2]"):
        with pytest.raises(ValueError):
            parse_reading(text, space, model="m", latency_s=0)


def test_schema_and_prompt_follow_the_space(space, parking):
    s = build_schema(space)
    assert set(space.scene_fields) <= set(s["properties"]) and s["required"] == list(s["properties"])
    assert s["properties"]["objects"]["items"]["properties"]["status"]["enum"] == STATUS_VALUES
    assert "weather_visible" in build_schema(parking)["properties"] and "weather_visible" not in s["properties"]
    assert "licence plates" in build_prompt(parking) and "chair" in build_prompt(space)


# ---- reader with a fake HTTP session
class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def ok_body(content=None, reasoning=None):
    msg = {"content": content if content is not None else json.dumps(GOOD)}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}]}


class Session:
    def __init__(self, *responses):
        self.responses, self.calls, self.last_json = list(responses), 0, None

    def post(self, url, json=None, timeout=None):
        self.last_json = json
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


def reader(session, retries=3, **kw):
    cfg = SceneReaderSettings(base_url="http://x", model="m", retries=retries, **kw)
    return LmStudioSceneReader(cfg, session=session, sleep=lambda s: None)


def test_reader_success_sends_schema_and_reasoning_effort(space):
    s = Session(Resp(200, ok_body()))
    r = reader(s).read(synthetic_image(), space)
    assert r.people_count == 3 and s.calls == 1
    assert s.last_json["reasoning_effort"] == "none"
    assert s.last_json["response_format"]["type"] == "json_schema"


def test_reader_falls_back_to_reasoning_content_when_content_is_empty(space):
    s = Session(Resp(200, ok_body(content="", reasoning=json.dumps(GOOD))))
    assert reader(s).read(synthetic_image(), space).people_count == 3


def test_reader_retries_transient_failures_then_succeeds(space):
    s = Session(requests.ConnectionError("down"), Resp(503), Resp(200, ok_body("garbage")), Resp(200, ok_body()))
    assert reader(s, retries=4).read(synthetic_image(), space).people_count == 3 and s.calls == 4


def test_reader_does_not_retry_client_errors(space):
    s = Session(Resp(400, text="bad request"))
    with pytest.raises(VLMError, match="HTTP 400"):
        reader(s).read(synthetic_image(), space)
    assert s.calls == 1


def test_reader_gives_up_with_vlmerror_after_retries(space):
    s = Session(requests.Timeout("slow"))
    with pytest.raises(VLMError, match="after 3 attempts"):
        reader(s).read(synthetic_image(), space)
    assert s.calls == 3


def test_reader_omits_reasoning_effort_when_unset(space):
    s = Session(Resp(200, ok_body()))
    reader(s, reasoning_effort=None).read(synthetic_image(), space)
    assert "reasoning_effort" not in s.last_json

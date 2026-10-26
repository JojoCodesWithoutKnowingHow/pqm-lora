"""The caption service without a GPU: tag selection, and the HTTP routes over fakes."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pqm_lora import models, serve, tags

ROOT = Path(__file__).resolve().parent.parent


def test_select_orders_by_score_and_splits_categories():
    labels = [("general", "9"), ("sensitive", "9"), ("pink_hair", "0"), ("solo", "0"),
              ("^_^", "0"), ("crona_(soul_eater)", "4"), ("faint", "0")]
    scores = [0.8, 0.2, 0.91, 0.97, 0.35, 0.88, 0.19]
    out = tags.select(scores, labels, 0.2)
    assert out["general"] == [["solo", 0.97], ["pink_hair", 0.91], ["^_^", 0.35]]
    assert out["character"] == [["crona_(soul_eater)", 0.88]]
    assert out["rating"] == [["general", 0.8], ["sensitive", 0.2]]


def test_select_keeps_the_models_order_on_a_tie():
    labels = [("b", "0"), ("a", "0")]
    assert tags.select([0.5, 0.5], labels, 0.1)["general"] == [["b", 0.5], ["a", 0.5]]


def test_select_refuses_a_mismatch():
    with pytest.raises(ValueError):
        tags.select([0.1], [("a", "0"), ("b", "0")], 0.1)


class FakeTagger:
    calls = 0

    def __init__(self):
        FakeTagger.calls += 1

    def tag(self, path, min_score):
        return {"general": [["solo", 0.9]], "character": [], "rating": [], "path": path,
                "min_score": min_score, "model": "fake"}


class FakeJoy:
    def caption(self, path, prompt, max_new_tokens):
        return {"sentence": "A figure stands.", "prompt": prompt, "tokens": max_new_tokens}


@pytest.fixture
def service():
    FakeTagger.calls = 0
    box = {}
    ready = threading.Event()

    def up(server):
        box["server"] = server
        ready.set()

    held = serve.Models(FakeTagger, FakeJoy)
    thread = threading.Thread(target=serve.serve, kwargs={"port": 0, "models_": held,
                                                          "ready": up}, daemon=True)
    thread.start()
    assert ready.wait(5)
    server = box["server"]
    yield "http://127.0.0.1:%d" % server.server_port, held
    server.shutdown()


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as answer:
            return answer.status, json.loads(answer.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_answers_before_any_model_loads(service):
    base, held = service
    status, body = call(base + "/health")
    assert status == 200 and body["ok"] and body["loaded"] == {"tagger": False, "joycaption": False}


def test_tag_loads_once_and_reports(service, tmp_path):
    base, held = service
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    for _ in range(3):
        status, body = call(base + "/tag", {"path": str(image), "min_score": 0.3})
        assert status == 200 and body["general"] == [["solo", 0.9]] and body["min_score"] == 0.3
    assert FakeTagger.calls == 1
    assert call(base + "/health")[1]["loaded"]["tagger"] is True


def test_caption_passes_the_prompt(service, tmp_path):
    base, _ = service
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    status, body = call(base + "/caption", {"path": str(image), "prompt": "Describe it."})
    assert status == 200 and body["sentence"] == "A figure stands."
    assert body["tokens"] == models.MAX_NEW_TOKENS


@pytest.mark.parametrize("body,said", [
    ({}, "absolute path"),
    ({"path": "relative.png"}, "not an absolute path"),
])
def test_bad_requests_are_400_with_a_reason(service, body, said):
    base, _ = service
    status, answer = call(base + "/tag", body)
    assert status == 400 and said in answer["error"]


def test_a_missing_image_is_400(service, tmp_path):
    base, _ = service
    status, answer = call(base + "/tag", {"path": str(tmp_path / "missing.png")})
    assert status == 400 and "no file" in answer["error"]


def test_a_caption_without_a_prompt_is_refused(service, tmp_path):
    base, _ = service
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    status, answer = call(base + "/caption", {"path": str(image)})
    assert status == 400 and "prompt" in answer["error"]


def test_a_model_failure_is_500(tmp_path):
    def broken():
        raise RuntimeError("CUDA out of memory")

    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    held = serve.Models(broken, FakeJoy)
    with pytest.raises(RuntimeError):
        serve.handle(held, "/tag", {"path": str(image)})


def test_pod_setup_is_lf_and_pins_every_requirement():
    script = (ROOT / "setup" / "pod_setup.sh").read_bytes()
    assert b"\r\n" not in script
    for line in (ROOT / "requirements-pod.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert "==" in line, line
    assert len(models.TAGGER_REVISION) == 40 and len(models.JOYCAPTION_REVISION) == 40

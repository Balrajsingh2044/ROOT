"""Run from the truwage-mvp folder:  python -m pytest -q"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.integrity_checker import integrity_checker
from app.nsqf_mapper import map_skills
from app.passport import build_passport, compute_hash
from app.wage_engine import predict_wage


# ---------- synthetic test videos ----------
def _write(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def _busy_frames(seed=1, n=360):
    rng = np.random.default_rng(seed)
    bg = cv2.resize(rng.integers(60, 180, (30, 40, 3), dtype=np.uint8), (320, 240), interpolation=cv2.INTER_CUBIC)
    pos, vel, frames = rng.uniform(40, 200, (4, 2)), rng.normal(0, 3, (4, 2)), []
    for _ in range(n):
        vel = np.clip(vel + rng.normal(0, 1.2, vel.shape), -8, 8)
        pos += vel
        pos[:, 0] %= 300
        pos[:, 1] %= 220
        f = bg.copy()
        for k, (x, y) in enumerate(pos):
            cv2.rectangle(f, (int(x), int(y)), (int(x) + 40, int(y) + 30), (30 + 50 * k, 200 - 40 * k, 90 + 30 * k), -1)
        frames.append(np.clip(f.astype(int) + rng.integers(-2, 3, f.shape), 0, 255).astype(np.uint8))
    return frames


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    frames = _busy_frames()
    _write(d / "genuine.mp4", frames)
    _write(d / "looped.mp4", [frames[i % 90] for i in range(len(frames))])
    _write(d / "frozen.mp4", [frames[0]] * len(frames))
    return d


# ---------- wage engine ----------
def test_wage_rises_with_skill():
    assert predict_wage(0.2, 3, 3, seed=1)["mean"] < predict_wage(0.9, 3, 3, seed=1)["mean"]


def test_wage_rises_with_complexity_and_demand():
    assert predict_wage(0.5, 1, 1, seed=1)["mean"] < predict_wage(0.5, 5, 5, seed=1)["mean"]


def test_wage_range_is_ordered_and_formatted():
    w = predict_wage(0.6, 3, "high", seed=1)
    assert w["min"] < w["max"]
    assert w["text"].count("\u20b9") == 2 and w["text"].endswith("/hr")
    assert w["confidence"] in {"High", "Medium", "Low"}


# ---------- integrity ----------
def test_integrity_passes_genuine(clips):
    assert integrity_checker.predict(str(clips / "genuine.mp4"))["is_authentic"]


def test_integrity_flags_frozen(clips):
    r = integrity_checker.predict(str(clips / "frozen.mp4"))
    assert not r["is_authentic"] and "still" in r["message"]


def test_integrity_flags_loop(clips):
    r = integrity_checker.predict(str(clips / "looped.mp4"))
    assert not r["is_authentic"] and "repeats" in r["message"]


def test_integrity_handles_missing_file(tmp_path):
    assert not integrity_checker.predict(str(tmp_path / "nope.mp4"))["is_authentic"]


# ---------- NSQF ----------
def test_nsqf_dropdown_and_alias():
    assert map_skills("", "plumbing")["trade"] == "Plumbing"
    assert map_skills("", "Carpenter")["trade"] == "Carpentry"


def test_nsqf_from_hindi_transcript():
    card = map_skills("मेरा काम पाइप लीकेज ठीक करना है", "")
    assert card["trade"] == "Plumbing" and card["source"] == "transcript"


def test_nsqf_unknown_is_unmapped():
    assert map_skills("", "astronaut")["nsqf_level"] is None


# ---------- passport ----------
FIELDS = dict(worker_name="Asha", trade="Plumbing", nsqf_level="NSQF Level 2 (illustrative)", score="0.71",
              integrity_status="PASS", wage_range="\u20b9240 - \u20b9270/hr", wage_confidence="High")


def test_passport_escapes_html_and_contains_hash(tmp_path):
    r = build_passport({**FIELDS, "worker_name": "<script>x</script>"}, "ab" * 32, tmp_path, "p1")
    page = r["path"].read_text(encoding="utf-8")
    assert "<script>" not in page and r["short_hash"] in page


def test_passport_hash_changes_with_video_or_fields(tmp_path):
    a = build_passport(FIELDS, "aa" * 32, tmp_path, "a")
    b = build_passport(FIELDS, "bb" * 32, tmp_path, "b")
    assert a["hash"] != b["hash"]
    assert compute_hash("aa" * 32, {**FIELDS, "score": "0.99", "issued": a["issued"]}) != \
        compute_hash("aa" * 32, {**FIELDS, "issued": a["issued"]})


# ---------- API ----------
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "run_aqa", lambda path: {"score": 0.72, "heatmap": "data:image/jpeg;base64,AAAA"})
    with TestClient(main.app) as c:
        yield c


def _post(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/api/assess", files={"file": (path.name, f, "video/mp4")}, data=form)


def test_api_genuine_video_passes(client, clips):
    r = _post(client, clips / "genuine.mp4", task_type="plumbing", complexity="4", demand="high", worker_name="Asha")
    body = r.json()
    assert r.status_code == 200 and body["deepfake_status"] == "PASS"
    assert body["score"] == 0.72 and body["wage_range_text"].count("\u20b9") == 2
    assert body["trade"] == "Plumbing" and body["verification_hash"]
    page = client.get(body["passport_url"])
    assert page.status_code == 200 and body["verification_hash"] in page.text


def test_api_wage_responds_to_complexity(client, clips):
    low = _post(client, clips / "genuine.mp4", complexity="1", demand="low").json()
    high = _post(client, clips / "genuine.mp4", complexity="5", demand="high").json()
    assert low["wage_max"] < high["wage_max"]


def test_api_looped_video_fails_without_wage(client, clips):
    body = _post(client, clips / "looped.mp4").json()
    assert body["deepfake_status"] == "FAIL" and body["wage_range_text"] == "N/A" and body["heatmap"] is None


def test_api_rejects_non_video_and_bad_complexity(client, clips):
    bad = client.post("/api/assess", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
    assert _post(client, clips / "genuine.mp4", complexity="9").status_code == 422


def test_api_returns_503_when_aqa_missing(monkeypatch, clips):
    monkeypatch.setattr(main, "run_aqa", None)
    with TestClient(main.app) as c:
        assert _post(c, clips / "genuine.mp4").status_code == 503


def test_api_leaves_no_temp_files(client, clips):
    _post(client, clips / "genuine.mp4")
    assert list(main.TEMP_DIR.glob("*")) == []


def test_get_wage_endpoint(client):
    r = client.post("/get_wage", json={"task_type": "plumbing", "complexity": 3, "location_demand": "high", "skill_score": 0.8})
    assert r.status_code == 200 and r.json()["estimated_wage_min"] < r.json()["estimated_wage_max"]

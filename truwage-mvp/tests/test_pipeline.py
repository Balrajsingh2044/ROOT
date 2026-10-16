"""Run from the truwage-mvp folder:  python -m pytest -q"""
from pathlib import Path

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
def temp_dir(tmp_path, monkeypatch):
    """Isolated upload folder, so leftovers from earlier runs cannot affect the cleanup tests."""
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(main, "TEMP_DIR", d)
    return d


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


def test_api_leaves_no_temp_files(client, clips, temp_dir):
    _post(client, clips / "genuine.mp4")
    assert list(temp_dir.glob("*")) == []


def test_get_wage_endpoint(client):
    r = client.post("/get_wage", json={"task_type": "plumbing", "complexity": 3, "location_demand": "high", "skill_score": 0.8})
    assert r.status_code == 200 and r.json()["estimated_wage_min"] < r.json()["estimated_wage_max"]


# ---------- voice note ----------
def _wav(path, seconds=1.0, rate=16000):
    import wave
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return path


def test_load_audio_decodes_wav_to_16k_mono(tmp_path):
    from app.transcriber import load_audio
    audio = load_audio(_wav(tmp_path / "tone.wav", seconds=1.0, rate=8000))   # 8 kHz input is resampled
    assert audio.dtype == np.float32 and abs(len(audio) - 16000) < 400
    assert 0.1 < float(np.abs(audio).max()) < 0.5


def test_load_audio_rejects_garbage(tmp_path):
    from app.transcriber import load_audio
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(RuntimeError):
        load_audio(bad)


def _post_with_audio(client, video, audio, **form):
    with open(video, "rb") as v, open(audio, "rb") as a:
        return client.post("/api/assess",
                           files={"file": (video.name, v, "video/mp4"), "audio": (audio.name, a, "audio/wav")},
                           data=form)


def test_api_voice_note_is_transcribed(client, clips, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio", lambda path, language: "मेरा काम पाइप लीकेज ठीक करना है")
    body = _post_with_audio(client, clips / "genuine.mp4", _wav(tmp_path / "v.wav"),
                            task_type="plumbing", language="hindi").json()
    assert body["deepfake_status"] == "PASS"
    assert body["transcript"] == "मेरा काम पाइप लीकेज ठीक करना है"
    assert body["voice_trade"] == "Plumbing" and body["transcript_note"] is None
    assert "पाइप" in body["matched_keywords"]


def test_api_voice_note_trade_mismatch_is_flagged(client, clips, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio", lambda path, language: "मैं लकड़ी का फर्नीचर बनाता हूँ")
    body = _post_with_audio(client, clips / "genuine.mp4", _wav(tmp_path / "v.wav"), task_type="plumbing").json()
    assert body["voice_trade"] == "Carpentry" and "selected trade is Plumbing" in body["transcript_note"]
    assert body["trade"] == "Plumbing"          # the selected trade still decides the skill card


def test_api_transcription_failure_does_not_break_assessment(client, clips, tmp_path, monkeypatch):
    def boom(path, language):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(main, "transcribe_audio", boom)
    body = _post_with_audio(client, clips / "genuine.mp4", _wav(tmp_path / "v.wav")).json()
    assert body["deepfake_status"] == "PASS" and body["transcript"] is None
    assert "model unavailable" in body["transcript_note"]


def test_api_rejects_bad_audio_type_and_cleans_up(client, clips, temp_dir):
    with open(clips / "genuine.mp4", "rb") as v:
        r = client.post("/api/assess", files={"file": ("g.mp4", v, "video/mp4"), "audio": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400
    assert list(temp_dir.glob("*")) == []


def test_api_without_audio_has_no_transcript(client, clips):
    body = _post(client, clips / "genuine.mp4").json()
    assert body["transcript"] is None and body["voice_trade"] is None


def test_api_audio_temp_files_are_removed(client, clips, temp_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio", lambda path, language: "")
    body = _post_with_audio(client, clips / "genuine.mp4", _wav(tmp_path / "v.wav")).json()
    assert body["transcript_note"] == "No speech was recognised in the audio."
    assert list(temp_dir.glob("*")) == []


# ---------- temp-file cleanup on Windows-style locks ----------
def test_remove_file_retries_when_the_file_is_briefly_locked(tmp_path, monkeypatch):
    import asyncio
    f = tmp_path / "locked.mp4"
    f.write_bytes(b"x")
    real_unlink, calls = Path.unlink, {"n": 0}

    def flaky_unlink(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("[WinError 32] file is being used by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    assert asyncio.run(main._remove_file(f, attempts=5, delay=0)) is True
    assert not f.exists() and calls["n"] == 3


def test_sweep_removes_only_stale_files(tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setattr(main, "TEMP_DIR", tmp_path)
    old, new = tmp_path / "old.mp4", tmp_path / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    main.sweep_temp_files(max_age_seconds=3600)
    assert not old.exists() and new.exists()

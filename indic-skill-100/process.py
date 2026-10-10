import re
import subprocess
import sys
from pathlib import Path

RAW = Path("raw_videos")
OUT_VIDEO = Path("processed/video_15fps")
OUT_AUDIO = Path("processed/audio")
OUT_VIDEO.mkdir(parents=True, exist_ok=True)
OUT_AUDIO.mkdir(parents=True, exist_ok=True)

EXTS = {".mp4", ".mov", ".mkv", ".webm"}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("   ffmpeg error:", r.stderr[-300:])
    return r.returncode == 0


def has_real_audio(path: Path) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if not probe.stdout.strip():
        return False
    vol = subprocess.run(
        ["ffmpeg", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(\S+) dB", vol.stderr)
    return bool(m) and float(m.group(1)) > -50


def process(video: Path):
    stem = video.stem
    out_v = OUT_VIDEO / f"{stem}.mp4"

    if out_v.exists():
        v_status = "video exists"
    else:
        ok_v = run([
            "ffmpeg", "-y", "-i", str(video),
            "-map", "0:v:0",
            "-vf", "fps=15,scale=-2:480",
            "-an",
            "-c:v", "libx264", "-crf", "28", "-preset", "fast",
            str(out_v),
        ])
        v_status = "video ok" if ok_v else "video FAILED"

    if has_real_audio(video):
        ok_a = run([
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(OUT_AUDIO / f"{stem}.wav"),
        ])
        a_status = "audio ok" if ok_a else "audio FAILED"
    else:
        a_status = "no audio"

    print(f"{stem}: {v_status}, {a_status}")


if __name__ == "__main__":
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit("No videos found in raw_videos/")
    print(f"Processing {len(files)} videos...")
    for f in files:
        process(f)
    print("Done.")
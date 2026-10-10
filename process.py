import subprocess
import sys
from pathlib import Path

RAW = Path("raw_videos")
OUT_VIDEO = Path("processed/video_15fps")
OUT_VIDEO.mkdir(parents=True, exist_ok=True)

EXTS = {".mp4", ".mov", ".mkv", ".webm"}

# Set to True later if you ever add clips with real speech.
EXTRACT_AUDIO = False
OUT_AUDIO = Path("processed/audio")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("   ffmpeg error:", r.stderr[-300:])
    return r.returncode == 0


def process(video: Path):
    stem = video.stem

    # 15 fps, 480p, first video stream only, audio dropped
    ok_v = run([
        "ffmpeg", "-y", "-i", str(video),
        "-map", "0:v:0",
        "-vf", "fps=15,scale=-2:480",
        "-an",
        "-c:v", "libx264", "-crf", "28", "-preset", "fast",
        str(OUT_VIDEO / f"{stem}.mp4"),
    ])

    if EXTRACT_AUDIO:
        OUT_AUDIO.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
             "-ar", "16000", "-c:a", "pcm_s16le",
             str(OUT_AUDIO / f"{stem}.wav")])

    print(f"{stem}: video {'ok' if ok_v else 'FAILED'}")


if __name__ == "__main__":
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit("No videos found in raw_videos/")
    print(f"Processing {len(files)} videos...")
    for f in files:
        process(f)
    print("Done.")
import csv
import subprocess
from pathlib import Path

VID = Path("processed/video_15fps")
AUD = Path("processed/audio")
RAW = Path("raw_videos")


def fps(p):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(p)],
        capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    return lines[0].split(",")[0].strip() if lines else ""


problems = 0
raw_names = {p.name for p in RAW.iterdir() if p.is_file()}
with open("metadata.csv", encoding="utf-8") as fh:
    meta = {r["filename"]: r for r in csv.DictReader(fh)}

for n in raw_names - set(meta):
    print("MISSING in metadata:", n); problems += 1
for n in set(meta) - raw_names:
    print("In metadata but no raw file:", n); problems += 1

for name, row in meta.items():
    stem = Path(name).stem
    v = VID / f"{stem}.mp4"
    if not v.exists():
        print("No processed video:", name); problems += 1
    elif fps(v) != "15/1":
        print(f"Wrong fps for {name}: {fps(v)}"); problems += 1
    if (row["has_audio"] == "yes") != (AUD / f"{stem}.wav").exists():
        print("Audio mismatch:", name); problems += 1
    if not row["task_description"] or not row["source_url"]:
        print("Incomplete metadata:", name); problems += 1

n_audio = sum(1 for r in meta.values() if r["has_audio"] == "yes")
print(f"\nChecked {len(meta)} clips ({n_audio} with audio), {problems} problem(s).")
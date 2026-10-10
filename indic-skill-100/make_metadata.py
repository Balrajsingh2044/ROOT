import csv
from pathlib import Path

RAW = Path("raw_videos")
AUD = Path("processed/audio")
EXTS = {".mp4", ".mov", ".mkv", ".webm"}

yt = {}
if Path("yt_sources.csv").exists():
    with open("yt_sources.csv", encoding="utf-8") as fh:
        yt = {r["filename"]: r["url"] for r in csv.DictReader(fh)}

rows = []
for f in sorted(p for p in RAW.iterdir() if p.suffix.lower() in EXTS):
    if f.name in yt:
        audio = (AUD / f"{f.stem}.wav").exists()
        rows.append({
            "filename": f.name,
            "trade": "unspecified",
            "task_description": "youtube short (fill in)",
            "source_site": "youtube",
            "source_url": yt[f.name],
            "license": "YouTube standard license (research use only)",
            "has_audio": "yes" if audio else "no",
            "language": "unknown" if audio else "none",
        })
    else:
        trade = f.stem.split("_")[0]
        rows.append({
            "filename": f.name,
            "trade": trade,
            "task_description": f"{trade} task (stock clip)",
            "source_site": "pexels",
            "source_url": "https://www.pexels.com/videos/",
            "license": "Pexels License",
            "has_audio": "no",
            "language": "none",
        })

with open("metadata.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} rows to metadata.csv")
import csv
from pathlib import Path

RAW = Path("raw_videos")
EXTS = {".mp4", ".mov", ".mkv", ".webm"}

rows = []
for f in sorted(p for p in RAW.iterdir() if p.suffix.lower() in EXTS):
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
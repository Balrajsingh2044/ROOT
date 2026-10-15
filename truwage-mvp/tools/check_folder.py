r"""Run the integrity check on every video in a folder and print a summary.

Usage (from the truwage-mvp folder):
    python tools\check_folder.py C:\path\to\indic-skill-100\processed\video_15fps
Writes integrity_report.csv next to where you run it.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.integrity_checker import integrity_checker  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def category(message):
    m = message.lower()
    if "still" in m:
        return "frozen/still"
    if "repeats" in m:
        return "looped"
    if "short" in m:
        return "too short"
    return "unreadable"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    folder = Path(sys.argv[1])
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS) if folder.is_dir() else []
    if not files:
        raise SystemExit(f"No videos found in {folder}")

    rows, reasons = [], Counter()
    for i, p in enumerate(files, 1):
        r = integrity_checker.predict(str(p))
        rows.append((p.name, r["is_authentic"], r["confidence_score"], r["message"]))
        if not r["is_authentic"]:
            reasons[category(r["message"])] += 1
        print(f"[{i}/{len(files)}] {'PASS' if r['is_authentic'] else 'FAIL'}  {p.name}", flush=True)

    with open("integrity_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "authentic", "confidence", "message"])
        w.writerows(rows)

    passed = sum(1 for r in rows if r[1])
    print("\n" + "=" * 60)
    print(f"{len(rows)} videos checked: {passed} passed, {len(rows) - passed} failed")
    for reason, n in reasons.items():
        print(f"  {reason}: {n}")
    for name, ok, _, msg in rows:
        if not ok:
            print(f"  FAIL {name}: {msg}")
    print("Full report: integrity_report.csv")


if __name__ == "__main__":
    main()

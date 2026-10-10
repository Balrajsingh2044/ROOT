import subprocess
import re
from pathlib import Path

for w in sorted(Path("processed/audio").glob("*.wav")):
    r = subprocess.run(
        ["ffmpeg", "-i", str(w), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(\S+) dB", r.stderr)
    mean = float(m.group(1)) if m else float("-inf")
    verdict = "SILENT" if mean < -50 else "has sound"
    print(f"{w.name}: mean {mean} dB -> {verdict}")
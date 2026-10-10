import csv
import re
import subprocess
from pathlib import Path

YTDLP = r".\yt-dlp.exe"      # standalone yt-dlp in the indic folder
LINKS = Path("audio_links.txt")
RAW = Path("raw_videos")
SRC = Path("yt_sources.csv")
COOKIES = []                 # if YouTube asks to sign in: ["--cookies-from-browser", "chrome"]

RAW.mkdir(exist_ok=True)

if not LINKS.exists():
    raise SystemExit(f"Not found: {LINKS}")

# unique links, order kept, blanks removed
urls = list(dict.fromkeys(
    l.strip() for l in LINKS.read_text(encoding="utf-8").splitlines() if l.strip()))
print(f"{len(urls)} unique links")

done = {}
if SRC.exists():
    with open(SRC, encoding="utf-8") as fh:
        done = {r["url"]: r["filename"] for r in csv.DictReader(fh)}


def next_num():
    nums = [int(m.group(1)) for p in RAW.glob("speech_*.mp4")
            if (m := re.match(r"speech_(\d+)\.mp4$", p.name))]
    return max(nums, default=0) + 1


new_file = not SRC.exists()
ok_count, failed = 0, []
with open(SRC, "a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if new_file:
        w.writerow(["filename", "url"])

    for i, url in enumerate(urls, 1):
        if url in done:
            print(f"[{i}/{len(urls)}] already downloaded: {url}")
            continue

        name = f"speech_{next_num():03d}"
        print(f"[{i}/{len(urls)}] {name}  {url}")
        r = subprocess.run([
            YTDLP, "--no-playlist",
            "-f", "bv*[height<=480]+ba/b[height<=480]",
            "--merge-output-format", "mp4",
            "-o", f"raw_videos/{name}.%(ext)s",
            *COOKIES, url])

        out = RAW / f"{name}.mp4"
        if r.returncode == 0 and out.exists():
            w.writerow([out.name, url])
            done[url] = out.name
            fh.flush()
            ok_count += 1
        else:
            failed.append(url)
            print("   FAILED")

print(f"\nDownloaded {ok_count}, failed {len(failed)}")
for u in failed:
    print("  FAILED:", u)
"""Make a LOOPED and a FROZEN version of a genuine clip, for testing the integrity check.

Usage:  python tools/make_tamper_clips.py path/to/genuine.mp4 demo_clips
Writes: demo_clips/looped.mp4 and demo_clips/frozen.mp4 (same length as the input).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2


def read_frames(path, max_frames=2000):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def find_ffmpeg():
    """imageio-ffmpeg bundles an ffmpeg binary; otherwise use ffmpeg from PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def to_h264(src, dst):
    """Re-encode to H.264 so the file also plays in a web browser (Streamlit preview)."""
    exe = find_ffmpeg()
    if exe is None:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", "-an", str(dst)]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and Path(dst).exists()


def write_video(path, frames, fps):
    path = Path(path)
    h, w = frames[0].shape[:2]
    tmp = path.with_name("_tmp_" + path.name)
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    if to_h264(tmp, path):
        tmp.unlink()
    else:
        tmp.replace(path)
        print(f"WARNING: could not convert {path.name} to H.264. It may not play in the browser preview. "
              "Run: pip install imageio-ffmpeg   and re-run this tool.")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_frames(src)
    if len(frames) < 30:
        raise SystemExit("Input video is too short.")
    total = len(frames)
    seg = max(int(fps * 3), 1)                      # repeat a 3-second segment
    start = total // 4 if total // 4 + seg <= total else 0   # avoid fade-ins at the very start
    looped = [frames[start + (i % seg)] for i in range(total)]
    frozen = [frames[total // 2]] * total           # a mid-clip frame, not a possibly black first frame
    write_video(out_dir / "looped.mp4", looped, fps)
    write_video(out_dir / "frozen.mp4", frozen, fps)
    print(f"Wrote {out_dir / 'looped.mp4'} and {out_dir / 'frozen.mp4'} ({total} frames each)")


if __name__ == "__main__":
    main()

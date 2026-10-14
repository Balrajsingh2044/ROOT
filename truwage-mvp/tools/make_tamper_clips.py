"""Make a LOOPED and a FROZEN version of a genuine clip, for testing the integrity check.

Usage:  python tools/make_tamper_clips.py path/to/genuine.mp4 demo_clips
Writes: demo_clips/looped.mp4 and demo_clips/frozen.mp4 (same length as the input).
"""
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


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_frames(src)
    if len(frames) < 30:
        raise SystemExit("Input video is too short.")
    total = len(frames)
    seg = max(int(fps * 3), 1)                      # repeat the first 3 seconds
    looped = [frames[i % seg] for i in range(total)]
    frozen = [frames[0]] * total
    write_video(out_dir / "looped.mp4", looped, fps)
    write_video(out_dir / "frozen.mp4", frozen, fps)
    print(f"Wrote {out_dir / 'looped.mp4'} and {out_dir / 'frozen.mp4'} ({total} frames each)")


if __name__ == "__main__":
    main()

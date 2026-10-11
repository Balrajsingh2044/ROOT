"""Animated Grad-CAM overlay (GIF / MP4) and clean JSON for the AQA API."""
import argparse
import base64
import io
import json
import subprocess
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs"

SIZE = 256        # width and height of the picture part
STRIP_H = 12      # timeline strip under the picture
BANNER_H = 22     # honesty banner under the strip
FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNERS = {
    "human_rating": "DEMO - human rating",
    "model": "EXPERIMENTAL - not validated",
    "placeholder": "DEMO - placeholder score",
}


# ---------------------------------------------------------------- image helpers
def center_crop(rgb, size=224):
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    rgb = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))))
    h, w = rgb.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(rgb[y:y + size, x:x + size])


def overlay_red(rgb, heat, alpha=0.65):
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    a = (alpha * heat)[..., None]
    return (rgb * (1 - a) + red * a).astype(np.uint8)


def read_frames_at(path, wanted):
    """Read only the requested frame indices. Returns ({index: RGB frame}, fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    need, last, got, i = set(wanted), max(wanted), {}, 0
    while i <= last:
        if i in need:
            ok, f = cap.read()
        else:
            ok, f = cap.grab(), None
        if not ok:
            break
        if f is not None:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return got, (fps if fps and fps > 1 else 15.0)


def interp_over_time(arr, keys, t):
    """Linear interpolation of arr (indexed by sampled frame) at frame position t."""
    j = int(np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 1))
    if j >= len(keys) - 1 or keys[j + 1] == keys[j]:
        return arr[j]
    w = (t - keys[j]) / (keys[j + 1] - keys[j])
    return (1 - w) * arr[j] + w * arr[j + 1]


def render_frame(frame_rgb, heat224, flagged, t_sec, flag_cols, frac, banner):
    img = center_crop(frame_rgb, SIZE)
    if flagged:
        heat = cv2.resize(heat224, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        img = overlay_red(img, np.clip(heat, 0, 1))
        cv2.rectangle(img, (0, 0), (SIZE - 1, SIZE - 1), (255, 0, 0), 4)
    cv2.putText(img, f"{t_sec:.1f}s", (8, 22), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    strip = np.full((STRIP_H, SIZE, 3), 60, np.uint8)
    strip[:, flag_cols] = (220, 40, 40)
    mx = int(round(frac * (SIZE - 1)))
    strip[:, max(0, mx - 1): mx + 2] = 255

    bar = np.zeros((BANNER_H, SIZE, 3), np.uint8)
    cv2.putText(bar, banner, (6, 16), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, strip, bar])


# ---------------------------------------------------------------- encoders
def encode_gif(frames, duration_ms, colors):
    pil = [Image.fromarray(f).quantize(colors=colors, dither=0) for f in frames]
    buf = io.BytesIO()
    pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:],
                duration=duration_ms, loop=0, optimize=False)
    return buf.getvalue()


def build_gif_bytes(frames, gif_fps, max_kb):
    """Try richer settings first, fall back to fewer frames/colours until it fits."""
    base_ms = int(round(1000 / gif_fps))
    data, sub = b"", frames
    for stride, colors in ((1, 128), (1, 64), (2, 64), (2, 32), (3, 32)):
        sub = frames[::stride]
        data = encode_gif(sub, base_ms * stride, colors)
        if len(data) <= max_kb * 1024:
            break
    return data, len(sub)


def write_mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(path)]
    r = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr.decode(errors="ignore")[-300:])


def segments(times, flags):
    """Consecutive flagged frames -> [[start_sec, end_sec], ...]."""
    segs, start = [], None
    for k, (t, f) in enumerate(zip(times, flags)):
        if f and start is None:
            start = t
        if not f and start is not None:
            segs.append([round(start, 2), round(times[k - 1], 2)])
            start = None
    if start is not None:
        segs.append([round(start, 2), round(times[-1], 2)])
    return segs


def pack_heatmap(gif_bytes, gif_path, fmt):
    b64 = base64.b64encode(gif_bytes).decode("ascii")
    if fmt == "path":
        return str(Path(gif_path).resolve())
    return f"data:image/gif;base64,{b64}" if fmt == "data-uri" else b64


# ---------------------------------------------------------------- main logic
def explain_animated(model, processor, path, poor_below=0.50, frame_thresh=0.6,
                     out_root=OUT_ROOT, heatmap_format="data-uri", human=None,
                     n_out=32, gif_fps=8, max_kb=1500, mp4=False):
    import gradcam_aqa as g

    path = Path(path)
    frames_raw, idxs = g.load_frames(path)
    if frames_raw is None:
        raise ValueError(f"unreadable video: {path}")
    with torch.enable_grad():
        score, heat, fp = g.gradcam(model, processor,
                                    [g.center_crop(f) for f in frames_raw])

    source = "model" if model.trained else "placeholder"
    if human and path.stem in human:
        score, source = human[path.stem], "human_rating"
    poor_clip = score < poor_below

    keys = np.asarray(idxs, dtype=float)
    lo, hi = int(idxs[0]), int(idxs[-1])
    wanted = sorted({int(i) for i in np.linspace(lo, hi, max(2, min(n_out, hi - lo + 1))).round()})
    got, vfps = read_frames_at(path, wanted)
    src = dict(zip((int(i) for i in idxs), frames_raw))
    src.update(got)
    out_idx = [i for i in wanted if i in src]
    if len(out_idx) < 2:
        out_idx = sorted(src)

    def is_flagged(t):
        return bool(poor_clip and float(interp_over_time(fp, keys, t)) >= frame_thresh)

    span = max(1, hi - lo)
    flag_cols = np.array([is_flagged(x) for x in np.linspace(lo, hi, SIZE)])
    banner = BANNERS[source]

    frames, flags, times = [], [], []
    for i in out_idx:
        fl = is_flagged(i)
        t_sec = i / vfps
        frames.append(render_frame(src[i], interp_over_time(heat, keys, i), fl, t_sec,
                                   flag_cols, float(np.clip((i - lo) / span, 0, 1)), banner))
        flags.append(fl)
        times.append(t_sec)

    gif_bytes, n_used = build_gif_bytes(frames, gif_fps, max_kb)
    out_dir = Path(out_root) / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "overlay.gif"
    gif_path.write_bytes(gif_bytes)
    mp4_path = None
    if mp4:
        mp4_path = out_dir / "overlay.mp4"
        write_mp4(frames, mp4_path, gif_fps)

    # ---- the API contract: exactly the two AssessResponse fields that are Ayansh's ----
    contract = {"score": round(float(score), 4),
                "heatmap": pack_heatmap(gif_bytes, gif_path, heatmap_format)}
    g.validate_contract(contract)

    # ---- extra information for humans, kept OUT of the contract ----
    details = {
        "video_id": path.stem,
        "score": round(float(score), 4),
        "score_source": source,
        "model_trained": bool(model.trained),
        "poor_below": poor_below,
        "poor_segments_sec": segments(times, flags),
        "gif": str(gif_path),
        "gif_kb": round(len(gif_bytes) / 1024),
        "gif_frames": n_used,
        "mp4": str(mp4_path) if mp4_path else None,
    }
    (out_dir / "result.json").write_text(json.dumps(contract), encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (out_dir / "preview.html").write_text(
        f'<body style="background:#222"><img src="{pack_heatmap(gif_bytes, gif_path, "data-uri")}"'
        f' style="width:512px"></body>', encoding="utf-8")
    return contract, details


_CACHE = {}
_LOCK = threading.Lock()


def warmup():
    """Load the model once (call at server start so the first request is not slow)."""
    import gradcam_aqa as g
    if "model" not in _CACHE:
        _CACHE["processor"] = g.VideoMAEImageProcessor.from_pretrained(g.MODEL)
        _CACHE["model"] = g.AQAModel().eval()


def assess_video(video_path, heatmap_format="data-uri", use_human_scores=False,
                 poor_below=0.50):
    """Entry point for the backend. Returns {'score': float, 'heatmap': str (GIF)}."""
    import gradcam_aqa as g
    with _LOCK:
        warmup()
        human = None
        if use_human_scores:
            if "human" not in _CACHE:
                _CACHE["human"] = g.load_human()
            human = _CACHE["human"]
        try:
            contract, _ = explain_animated(_CACHE["model"], _CACHE["processor"], video_path,
                                           poor_below=poor_below,
                                           heatmap_format=heatmap_format, human=human)
            return contract
        except Exception:
            traceback.print_exc()
            print("WARNING: animated overlay failed, falling back to the static contact sheet")
            return g.assess_video(video_path, heatmap_format=heatmap_format,
                                  use_human_scores=use_human_scores)


def main():
    import gradcam_aqa as g
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--poor-below", type=float, default=0.50)
    ap.add_argument("--frame-thresh", type=float, default=0.6)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--heatmap-format", default="data-uri",
                    choices=["data-uri", "base64", "path"])
    ap.add_argument("--human-scores", action="store_true")
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-kb", type=int, default=1500)
    args = ap.parse_args()

    paths = [Path(p) for p in args.videos]
    if args.all:
        paths = sorted(g.VID.glob("*.mp4"))
        if args.limit:
            paths = paths[:args.limit]
    if not paths:
        raise SystemExit("Give a video path, or use --all")

    warmup()
    human = g.load_human() if args.human_scores else None
    model, processor = _CACHE["model"], _CACHE["processor"]
    if human:
        print("NOTE: score = averaged HUMAN rating. Heatmaps come from the model and show the mechanism only.\n")
    elif not model.trained:
        print("NOTE: no trained head found. Scores and heatmaps are a MOCK demo.\n")
    else:
        print("NOTE: weak experimental head loaded (not validated). Demo use only.\n")

    for p in paths:
        try:
            contract, d = explain_animated(
                model, processor, p, args.poor_below, args.frame_thresh, args.out,
                args.heatmap_format, human, args.frames, args.gif_fps, args.max_kb, args.mp4)
        except ValueError as e:
            print("skip:", e)
            continue
        shown = {**contract, "heatmap": contract["heatmap"][:45] + "..."}
        print(json.dumps(shown), f"[{d['score_source']}]")
        print(f"  poor segments (s): {d['poor_segments_sec']}  |  GIF {d['gif_kb']} KB, "
              f"{d['gif_frames']} frames -> {d['gif']}")


if __name__ == "__main__":
    main()
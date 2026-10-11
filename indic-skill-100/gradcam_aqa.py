import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel

MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
HERE = Path(__file__).resolve().parent
HEAD_PATH = HERE / "features" / "aqa_head.pt"
HUMAN_SCORES = HERE / "scores.csv"
VID = Path("processed/video_15fps")
NUM_FRAMES = 16
FPS = 15


class AQAModel(nn.Module):
    """VideoMAE backbone + linear scoring head. Output score is 0-1."""

    def __init__(self):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(MODEL)
        torch.manual_seed(0)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.trained = HEAD_PATH.exists()
        if self.trained:
            self.head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))

    def forward(self, pixel_values):
        h = self.backbone(pixel_values=pixel_values).last_hidden_state
        return torch.sigmoid(self.head(h.mean(dim=1))).squeeze(-1)


def load_human(path=HUMAN_SCORES):
    """Averaged human checklist ratings, keyed by clip name, on a 0-1 scale."""
    d = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["score"].strip():
                v = float(r["score"])
                d[Path(r["filename"].strip()).stem] = v / 100 if v > 1 else v
    return d


def load_frames(path, n=NUM_FRAMES):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, None
    idxs = np.linspace(0, total - 1, n).astype(int)
    need, kept, i = set(idxs.tolist()), {}, 0
    while i <= int(idxs.max()):
        ok, f = cap.read()
        if not ok:
            break
        if i in need:
            kept[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    used = [int(j) for j in idxs if j in kept]
    frames = [kept[j] for j in used]
    if not frames:
        return None, None
    while len(frames) < n:
        frames.append(frames[-1])
        used.append(used[-1])
    return frames, used


def center_crop(rgb, size=224):
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    rgb = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))))
    h, w = rgb.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return rgb[y:y + size, x:x + size]


def gradcam(model, processor, frames):
    """Return (score 0-1, heat[16,224,224] in 0-1, frame_poorness[16] in 0-1)."""
    store = {}

    def hook(_, __, out):
        act = out[0] if isinstance(out, tuple) else out
        act.retain_grad()
        store["act"] = act

    handle = model.backbone.encoder.layer[-1].register_forward_hook(hook)
    px = processor(frames, return_tensors="pt")["pixel_values"]
    model.zero_grad()
    score = model(px)
    (-score.sum()).backward()          # explain what pulls the score DOWN
    handle.remove()

    act = store["act"].detach()[0]
    grad = store["act"].grad.detach()[0]
    cam = F.relu((act * grad.mean(dim=0)).sum(dim=1))
    assert cam.numel() == 8 * 14 * 14, f"unexpected token count {cam.numel()}"
    cam = cam.reshape(8, 14, 14)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam16 = cam.repeat_interleave(2, dim=0)               # 8 time steps -> 16 frames
    heat = F.interpolate(cam16[:, None], size=(224, 224),
                         mode="bilinear", align_corners=False)[:, 0]
    fp = cam16.mean(dim=(1, 2))
    fp = (fp - fp.min()) / (fp.max() - fp.min() + 1e-8)
    return float(score.item()), heat.numpy(), fp.numpy()


def overlay_red(rgb, heat, alpha=0.65):
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    a = (alpha * heat)[..., None]
    return (rgb * (1 - a) + red * a).astype(np.uint8)


def encode_heatmap(sheet_rgb, fmt, sheet_path):
    """fmt: data-uri | base64 | path"""
    if fmt == "path":
        return str(Path(sheet_path).resolve())
    small = cv2.resize(sheet_rgb, (640, 640), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}" if fmt == "data-uri" else b64


def validate_contract(payload):
    """Validate against Shreyansh's AssessResponse. The other 3 fields get dummies."""
    root = Path(__file__).resolve().parent
    for p in (Path.cwd(), root, root.parent):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from schemas import AssessResponse
    except ImportError:
        print("WARNING: schemas.py not found, contract NOT validated")
        return False
    AssessResponse(deepfake_status="PASS", wage_range_text="",
                   audio_filepath="", **payload)
    return True


def explain_video(model, processor, path, poor_below=0.5, frame_thresh=0.6,
                  out_root="explain", heatmap_format="data-uri", human=None):
    path = Path(path)
    frames_raw, idxs = load_frames(path)
    if frames_raw is None:
        raise ValueError(f"unreadable video: {path}")
    frames = [center_crop(f) for f in frames_raw]
    score, heat, fp = gradcam(model, processor, frames)

    source = "model" if model.trained else "placeholder"
    if human and path.stem in human:
        score = human[path.stem]
        source = "human_rating"

    flagged = [i for i in range(NUM_FRAMES)
               if score < poor_below and fp[i] >= frame_thresh]

    out_dir = Path(out_root) / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    for i, f in enumerate(frames):
        t = overlay_red(f, heat[i]) if i in flagged else f.copy()
        if i in flagged:
            cv2.rectangle(t, (0, 0), (223, 223), (255, 0, 0), 4)
            cv2.imwrite(str(out_dir / f"poor_frame_{idxs[i]:04d}.png"),
                        cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
        cv2.putText(t, f"{idxs[i] / FPS:.1f}s", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])

    banner = {"human_rating": "DEMO - score is a human rating; heatmap shows mechanism only",
              "model": "EXPERIMENTAL - weak model, not validated",
              "placeholder": "DEMO - placeholder score, not validated"}[source]
    h_s = sheet.shape[0]
    cv2.rectangle(sheet, (0, h_s - 26), (sheet.shape[1], h_s), (0, 0, 0), -1)
    cv2.putText(sheet, banner, (8, h_s - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)

    sheet_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # ---- the API contract: exactly the two AssessResponse fields that are Ayansh's ----
    contract = {
        "score": round(float(score), 4),
        "heatmap": encode_heatmap(sheet, heatmap_format, sheet_path),
    }
    validate_contract(contract)

    # ---- extra info for humans, kept OUT of the contract ----
    details = {
        "video_id": path.stem,
        "score": round(float(score), 4),
        "score_source": source,
        "poor_below": poor_below,
        "poor_frames": [int(idxs[i]) for i in flagged],
        "poor_timestamps_sec": [round(idxs[i] / FPS, 2) for i in flagged],
        "model_trained": bool(model.trained),
    }
    (out_dir / "result.json").write_text(json.dumps(contract), encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return contract, details


_CACHE = {}


def assess_video(video_path, heatmap_format="data-uri", use_human_scores=False):
    """Importable entry point for the backend. Returns {'score': float, 'heatmap': str}."""
    if "model" not in _CACHE:
        _CACHE["processor"] = VideoMAEImageProcessor.from_pretrained(MODEL)
        _CACHE["model"] = AQAModel().eval()
    if use_human_scores and "human" not in _CACHE:
        _CACHE["human"] = load_human()
    contract, _ = explain_video(_CACHE["model"], _CACHE["processor"], video_path,
                                heatmap_format=heatmap_format,
                                human=_CACHE.get("human") if use_human_scores else None)
    return contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--poor-below", type=float, default=0.5)
    ap.add_argument("--frame-thresh", type=float, default=0.6)
    ap.add_argument("--out", default="explain")
    ap.add_argument("--heatmap-format", default="data-uri",
                    choices=["data-uri", "base64", "path"])
    ap.add_argument("--human-scores", action="store_true",
                    help="use the averaged human ratings in scores.csv as the score")
    args = ap.parse_args()

    paths = [Path(p) for p in args.videos]
    if args.all:
        paths = sorted(VID.glob("*.mp4"))
        if args.limit:
            paths = paths[:args.limit]
    if not paths:
        raise SystemExit("Give a video path, or use --all")

    human = load_human() if args.human_scores else None
    processor = VideoMAEImageProcessor.from_pretrained(MODEL)
    model = AQAModel().eval()
    print(f"MODEL = {MODEL}\nHEAD  = {HEAD_PATH.resolve()}  (exists: {HEAD_PATH.exists()})")
    if human:
        print("NOTE: score = averaged HUMAN rating (not model output). "
              "Heatmaps are from an untrained head and show the mechanism only.\n")
    elif not model.trained:
        print("NOTE: no trained scoring head (features/aqa_head.pt). "
              "Scores and heatmaps are a MOCK demo.\n")
    else:
        print("NOTE: weak experimental head loaded (not validated). Demo use only.\n")

    for p in paths:
        try:
            contract, d = explain_video(model, processor, p, args.poor_below,
                                        args.frame_thresh, args.out,
                                        args.heatmap_format, human)
        except ValueError as e:
            print("skip:", e)
            continue
        shown = {**contract, "heatmap": contract["heatmap"][:50] + "..."}
        print(json.dumps(shown), f"[{d['score_source']}]")
        print(f"  poor frames: {d['poor_frames']}  at {d['poor_timestamps_sec']} s")


if __name__ == "__main__":
    main()
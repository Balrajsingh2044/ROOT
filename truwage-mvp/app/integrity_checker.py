"""Video integrity check (heuristic).

Flags the two cheap fraud patterns the project cares about:
  1. FROZEN video : a still image or paused clip passed off as work footage.
  2. LOOPED video : a short recording repeated to fake a longer task.

HONEST SCOPE: this is a motion/repetition heuristic. It does NOT detect AI-generated video
or face-swap deepfakes. Tune the constants below on your own clips.
"""
import cv2
import numpy as np

TARGET_FPS = 10.0        # sampling rate used for the checks
MAX_SAMPLES = 320        # about 32 s of video at TARGET_FPS
MIN_SECONDS = 3.0        # shorter clips are rejected
SIZE = (64, 64)

# Frozen-video check
PIXEL_CHANGE = 12        # gray-level change that counts as "this pixel moved"
MOTION_FRACTION = 0.004  # a frame pair "has motion" if this share of pixels moved
MIN_MOVING_PAIRS = 0.15  # at least this share of frame pairs must have motion

# Loop check
LOOP_MIN_LAG_S = 2.0     # shortest repeat period we look for
LOOP_MIN_SAMPLES = 50    # need at least this many samples (about 5 s) to test for loops
LOOP_DIP = 0.7           # at the repeat lag, frames differ by less than this x the typical difference
LOOP_HIGH = 0.90         # ...for at least this share of the frame pairs at that lag


def _sample_frames(path):
    """Return (frames[n, 64, 64] float32 gray, effective_fps) or (None, 0.0) if unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None, 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1:
            fps = 30.0
        step = max(1, int(round(fps / TARGET_FPS)))
        frames, i = [], 0
        while len(frames) < MAX_SAMPLES:
            if i % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, SIZE, interpolation=cv2.INTER_AREA)
                frames.append(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32))
            else:
                if not cap.grab():
                    break
            i += 1
        if not frames:
            return None, 0.0
        return np.stack(frames), fps / step
    finally:
        cap.release()


def _moving_pair_share(frames):
    diffs = np.abs(np.diff(frames, axis=0))
    moved = (diffs > PIXEL_CHANGE).mean(axis=(1, 2))
    return float((moved > MOTION_FRACTION).mean())


def _loop_check(frames, eff_fps):
    """Return (is_loop, period_seconds, strength).

    A loop shows up as a sharp dip in frame difference at exactly one lag (the repeat
    period), compared with the typical difference over all lags in the same clip.
    """
    n = len(frames)
    if n < LOOP_MIN_SAMPLES:
        return False, 0.0, 0.0
    min_lag = int(round(LOOP_MIN_LAG_S * eff_fps))
    max_lag = n // 2
    if max_lag <= min_lag:
        return False, 0.0, 0.0
    lags = list(range(min_lag, max_lag + 1))
    medians = np.array([float(np.median(np.abs(frames[lag:] - frames[:-lag]).mean(axis=(1, 2))))
                        for lag in lags])
    typical = float(np.median(medians))
    if typical <= 0:
        return False, 0.0, 0.0
    best_share = 0.0
    for idx in np.argsort(lags):                      # shortest repeat period first
        if medians[idx] >= LOOP_DIP * typical:
            continue
        lag = lags[idx]
        d = np.abs(frames[lag:] - frames[:-lag]).mean(axis=(1, 2))
        share = float((d < LOOP_DIP * typical).mean())
        best_share = max(best_share, share)
        if share >= LOOP_HIGH:
            return True, lag / eff_fps, share
    return False, 0.0, best_share


class IntegrityChecker:
    def predict(self, video_path):
        frames, eff_fps = _sample_frames(video_path)
        if frames is None:
            return self._result(False, 0.0, "Could not read the video file.")
        duration = len(frames) / eff_fps
        if duration < MIN_SECONDS:
            return self._result(False, 0.1, f"Video is too short ({duration:.1f}s). Upload at least {MIN_SECONDS:.0f} seconds of work.")

        moving = _moving_pair_share(frames)
        if moving < MIN_MOVING_PAIRS:
            return self._result(False, 0.05, "Video is almost completely still. It looks like a frozen frame or a paused clip.",
                                moving_share=moving)

        is_loop, period, strength = _loop_check(frames, eff_fps)
        if is_loop:
            return self._result(False, 0.2, f"The video repeats itself about every {period:.1f} seconds. It looks like a looped clip.",
                                moving_share=moving, loop_period_s=period)

        confidence = round(min(0.99, 0.6 + 0.4 * min(1.0, moving * 2)), 2)
        return self._result(True, confidence, "Video passed the integrity check (no freeze or loop found).",
                            moving_share=moving)

    @staticmethod
    def _result(ok, confidence, message, **details):
        return {"is_authentic": ok, "confidence_score": round(float(confidence), 4),
                "message": message, "details": details}


integrity_checker = IntegrityChecker()

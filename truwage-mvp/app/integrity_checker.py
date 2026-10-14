import cv2
import numpy as np

def check_integrity_heuristics(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        return False, "Video too short"
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret2, frame2 = cap.read()
    if ret1 and ret2:
        diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
        if diff < 15.0:
            return False, "Potential looped video"
    cap.release()
    return True, "Integrity verified"
from gradcam_aqa import assess_video
from schemas import AssessResponse

out = assess_video("processed/video_15fps/plumbing_001.mp4")
print("keys:", list(out), "| score type:", type(out["score"]).__name__, "| heatmap chars:", len(out["heatmap"]))
AssessResponse(deepfake_status="PASS", wage_range_text="Rs 0", audio_filepath="", **out)
print("contract OK")
import json
import os

# Calculate path: from 'app/' go up two levels to 'ROOT/', then down to 'indic-skill-100/features/'
FEATURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../indic-skill-100/features"))

def load_contrastive_results():
    """Loads Ayansh's pre-computed AQA results."""
    json_path = os.path.join(FEATURES_DIR, "contrastive_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

def get_demo_aqa_score(video_filename: str):
    results = load_contrastive_results()
    default = {"score": 0.88}
    if not results:
        return default
    stem = os.path.splitext(video_filename)[0]
    if isinstance(results, dict):
        return results.get(stem, results)
    if isinstance(results, list):
        for r in results:
            if stem in str(r.get("video", r.get("filename", ""))):
                return r
        return results[0] if results else default
    return default
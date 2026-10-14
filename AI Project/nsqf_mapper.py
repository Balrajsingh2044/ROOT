import json
import os

def map_skills(transcript: str, trade_dropdown: str):
    nsqf_db = {
        "plumbing": {"level": 2, "tasks": ["pipe fitting", "leak repair"], "keywords": ["paip", "nal", "plumb", "प्लंबिंग"]},
        "electrical": {"level": 3, "tasks": ["wiring", "socket repair"], "keywords": ["wire", "bijli", "socket", "वायरिंग"]},
        "carpentry": {"level": 2, "tasks": ["wood cutting", "furniture assembly"], "keywords": ["lakdi", "furniture", "badhai", "लकड़ी"]},
        "welding": {"level": 3, "tasks": ["metal joining", "grinding"], "keywords": ["weld", "metal", "loha", "वेल्डिंग"]},
        "tailoring": {"level": 2, "tasks": ["stitching", "measurement"], "keywords": ["silai", "kapda", "darji", "सिलाई"]}
    }
    transcript_lower = transcript.lower()
    trade = trade_dropdown.lower() if trade_dropdown and trade_dropdown.lower() in nsqf_db else "general_labor"
    if trade == "general_labor":
        for t, data in nsqf_db.items():
            if any(kw in transcript_lower for kw in data["keywords"]):
                trade = t; break
    if trade == "general_labor":
        return {"trade": "General Labor", "level": "Illustrative Level 1", "matched_tasks": ["Basic assistance"]}
    data = nsqf_db[trade]
    return {"trade": trade.capitalize(), "level": f"NSQF Level {data['level']} (Illustrative)", "matched_tasks": data["tasks"]}
"""Keyword-based skill mapping to an NSQF-style skill card.

HONEST SCOPE: this matches words, it does not understand language (not zero-shot NLP) and
the NSQF levels in nsqf_skills.json are illustrative placeholders.
"""
import json
import re
from pathlib import Path

SKILLS_PATH = Path(__file__).resolve().parent / "nsqf_skills.json"
DISCLAIMER = ("Keyword-based mapping. NSQF levels are illustrative; verify against the official "
              "NSDC Qualification Packs before relying on them.")
ALIASES = {"carpenter": "carpentry", "plumber": "plumbing", "electrician": "electrical",
           "welder": "welding", "tailor": "tailoring", "painter": "painting", "mason": "masonry"}

_DB = None


def _db():
    global _DB
    if _DB is None:
        with open(SKILLS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _DB = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _DB


def trades():
    return list(_db().keys())


def _tokens(text):
    # Split on whitespace and punctuation only, so Devanagari words are not broken apart.
    return re.findall(r"[^\s,.;:!?()\"'\u0964]+", (text or "").lower())


def _matched_keywords(tokens, keywords):
    found = []
    for kw in keywords:
        k = kw.lower()
        if any(t == k or (len(k) >= 4 and t.startswith(k)) for t in tokens):
            found.append(kw)
    return found


def _normalise_trade(hint):
    h = (hint or "").strip().lower()
    h = ALIASES.get(h, h)
    return h if h in _db() else None


def map_skills(transcript="", trade_hint=""):
    """Return a skill card for the chosen trade (dropdown) or the best match in a transcript."""
    db = _db()
    tokens = _tokens(transcript)
    hits = {k: _matched_keywords(tokens, v["keywords"]) for k, v in db.items()}

    key, source = _normalise_trade(trade_hint), "dropdown"
    if key is None and any(hits.values()):
        key, source = max(hits, key=lambda k: len(hits[k])), "transcript"
    if key is None:
        return {"trade": "General labour", "trade_key": "general", "nsqf_level": None,
                "level_label": "Unmapped (no trade given)", "key_tasks": [],
                "matched_keywords": [], "source": "none", "disclaimer": DISCLAIMER}

    entry = db[key]
    return {"trade": key.capitalize(), "trade_key": key, "nsqf_level": entry["level"],
            "level_label": f"NSQF Level {entry['level']} (illustrative)",
            "key_tasks": entry["tasks"], "matched_keywords": hits[key],
            "source": source, "disclaimer": DISCLAIMER}

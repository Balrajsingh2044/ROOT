"""Digital Skill Passport: a one-page HTML record with a verification hash and QR code.

HONEST SCOPE: the hash makes the record tamper-EVIDENT (anyone holding the original video
and the listed fields can recompute it). It is not a digital signature and there is no
verification server yet.
"""
import base64
import hashlib
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_hash(video_sha256, fields):
    payload = json.dumps({"video_sha256": video_sha256, **fields}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _qr_data_uri(text):
    import qrcode  # imported here so the rest of the API still works if qrcode is missing
    buf = io.BytesIO()
    qrcode.make(text).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_passport(fields, video_sha256, out_dir, passport_id):
    """fields: worker_name, trade, nsqf_level, score, integrity_status, wage_range, wage_confidence."""
    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    record = {k: str(v) for k, v in fields.items()}
    record["issued"] = issued
    digest = compute_hash(video_sha256, record)
    short = digest[:16]

    e = lambda key: html.escape(record.get(key, ""))
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>TruWage Skill Passport</title>
<style>
 body {{ font-family: Arial, sans-serif; background:#f3f4f6; margin:0; padding:24px; }}
 .card {{ max-width:520px; margin:auto; background:#fff; border:2px solid #1f2937; border-radius:10px; padding:24px; }}
 h1 {{ margin:0 0 4px; font-size:22px; color:#111827; }}
 .sub {{ color:#6b7280; font-size:13px; margin-bottom:16px; }}
 table {{ width:100%; border-collapse:collapse; }}
 td {{ padding:7px 4px; border-bottom:1px solid #e5e7eb; font-size:15px; }}
 td:first-child {{ color:#6b7280; width:42%; }}
 .hash {{ font-family:monospace; font-size:12px; word-break:break-all; color:#374151; }}
 .note {{ font-size:11px; color:#6b7280; margin-top:12px; }}
 .qr {{ text-align:center; margin-top:14px; }}
</style></head><body><div class="card">
<h1>TruWage Digital Skill Passport</h1>
<div class="sub">Issued {e("issued")}</div>
<table>
<tr><td>Worker</td><td>{e("worker_name")}</td></tr>
<tr><td>Trade</td><td>{e("trade")}</td></tr>
<tr><td>NSQF level</td><td>{e("nsqf_level")}</td></tr>
<tr><td>Technique score</td><td>{e("score")} / 1.00</td></tr>
<tr><td>Integrity check</td><td>{e("integrity_status")}</td></tr>
<tr><td>Fair wage range</td><td>{e("wage_range")} (model confidence: {e("wage_confidence")})</td></tr>
</table>
<div class="qr"><img src="{_qr_data_uri('TRUWAGE-VERIFY:' + short)}" width="120" height="120" alt="verification QR"></div>
<div class="hash">Verification hash: {short}<br>Full SHA-256: {digest}</div>
<div class="note">Tamper-evident record only (SHA-256 of the original video plus the fields above).
Not a digitally signed credential. Wage figures come from a model trained on synthetic data;
NSQF levels are illustrative.</div>
</div></body></html>"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{passport_id}.html"
    path.write_text(page, encoding="utf-8")
    return {"path": path, "hash": digest, "short_hash": short, "issued": issued}

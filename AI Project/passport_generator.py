import qrcode
import hashlib
import datetime

def generate_passport_html(data: dict):
    hash_input = f"{data['filename']}_{data['score']}_{data['wage_range']}"
    verification_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    qr = qrcode.make(f"TRUWAGE-VERIFY:{verification_hash}")
    qr.save("passport_qr.png")
    html = f"""
    <div style="border: 2px solid #333; padding: 20px; font-family: Arial; max-width: 500px;">
        <h2 style="color: #2c3e50; text-align: center;">🛠️ Digital Skill Passport</h2>
        <p><strong>Trade:</strong> {data['trade']}</p>
        <p><strong>NSQF Level:</strong> {data['nsqf_level']}</p>
        <p><strong>Technique Score:</strong> {data['score']}/1.0</p>
        <p><strong>Integrity:</strong> {data['integrity_status']}</p>
        <p><strong>Fair Wage Range:</strong> {data['wage_range']}</p>
        <p><strong>Date:</strong> {datetime.date.today()}</p>
        <hr>
        <p style="font-size: 12px; color: #666;">Verification Hash: {verification_hash}</p>
        <div style="text-align: center;"><img src="passport_qr.png" width="100"></div>
    </div>
    """
    with open("digital_passport.html", "w", encoding="utf-8") as f:
        f.write(html)
    return "digital_passport.html"
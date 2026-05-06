# ==============================================================
# SmartMart CV Suite – Module 2: QR Generation & Multi‑Scanner
# (Fixed: missing libzbar)
# ==============================================================

# Install the missing system library for pyzbar
!apt-get install -y libzbar0 > /dev/null 2>&1
!pip install -q gradio qrcode[pil] opencv-python pandas pyzbar

import gradio as gr
import qrcode
import cv2
import numpy as np
import pandas as pd
import os, json, io, base64
from datetime import datetime, timedelta
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

# ---------- 1. Database setup ----------
DB_FILE = "/content/product_db.csv"
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=["product_id","name","shelf","expiry_date","selling_price","status"]).to_csv(DB_FILE, index=False)

def load_db():
    return pd.read_csv(DB_FILE)

def save_db(df):
    df.to_csv(DB_FILE, index=False)

# ---------- 2. QR Generation ----------
def generate_qr(product_id, name, shelf, expiry_date, selling_price, status):
    if not product_id.strip():
        return None, "❌ Product ID cannot be empty.", None

    data_dict = {
        "product_id": product_id.strip(),
        "name": name.strip(),
        "shelf": shelf.strip(),
        "expiry_date": expiry_date,
        "selling_price": float(selling_price),
        "status": status
    }
    json_data = json.dumps(data_dict)

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(json_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    qr_filename = f"/content/qr_{product_id.strip()}.png"
    img.save(qr_filename)

    # Update CSV database
    db = load_db()
    db = db[db["product_id"] != product_id.strip()]
    new_row = pd.DataFrame([data_dict])
    db = pd.concat([db, new_row], ignore_index=True)
    save_db(db)

    return img, f"✅ QR for '{product_id.strip()}' generated and stored in database.", qr_filename

# ---------- 3. Robust QR Decoder ----------
def decode_qr(image_path):
    """Try both OpenCV and pyzbar to decode a QR code. Returns decoded data string or None."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return None

    # --- Method 1: OpenCV QRCodeDetector ---
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img_cv)
    if data:
        return data

    # --- Method 2: pyzbar with preprocessing ---
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle varied lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Try pyzbar on original, grayscale, and thresholded images
    try:
        for im in [img_cv, gray, thresh]:
            decoded = pyzbar_decode(im)
            if decoded:
                return decoded[0].data.decode('utf-8')
    except Exception:
        pass
    return None

# ---------- 4. Scanner logic ----------
def scan_qr(uploaded_file, mode):
    if uploaded_file is None:
        return "⚠️ Please upload a QR image.", None

    decoded_data = decode_qr(uploaded_file)
    if not decoded_data:
        return "❌ No QR code found. Please upload a clear image of the QR.", None

    try:
        product = json.loads(decoded_data)
    except:
        return "❌ QR content is not valid product data.", None

    product_id = product.get("product_id", "")
    db = load_db()
    match = db[db["product_id"] == product_id]
    if match.empty:
        return f"⚠️ Product ID '{product_id}' not found in database. Generate its QR first.", None

    record = match.iloc[0].to_dict()

    # Expiry check
    expiry_alert = ""
    try:
        expiry_date = datetime.strptime(record["expiry_date"], "%Y-%m-%d")
        days_left = (expiry_date - datetime.today()).days
        if days_left < 0:
            expiry_alert = "🚨 EXPIRED – Remove immediately!"
        elif days_left <= 3:
            expiry_alert = f"⚠️ Expires in {days_left} days – Prioritize sale."
    except:
        pass

    if mode == "Shelf location":
        output = (
            f"📦 Product: {record['name']}\n"
            f"🏷️  ID: {product_id}\n"
            f"📍 Place on shelf: {record['shelf']}\n"
            f"📅 Expiry: {record['expiry_date']} {expiry_alert}"
        )
        return output, None

    elif mode == "POS (Billing)":
        if record["status"] == "billed":
            return f"❌ Product '{record['name']}' is already billed.", None
        # Mark as billed
        db.loc[db["product_id"] == product_id, "status"] = "billed"
        save_db(db)
        output = (
            f"🧾 BILLED: {record['name']}\n"
            f"💰 Amount: ${record['selling_price']:.2f}\n"
            f"📅 Expiry: {record['expiry_date']} {expiry_alert}\n"
            f"Status updated to 'billed'."
        )
        return output, None

    elif mode == "Gate (Exit check)":
        if record["status"] == "billed":
            output = f"✅ Gate clear – {record['name']} is billed."
        else:
            output = f"🚨 ALERT! {record['name']} is NOT billed. Do not let it pass!"
        return output + f"\n📅 Expiry: {record['expiry_date']} {expiry_alert}", None

    return "Invalid mode.", None

# ---------- 5. Database viewer ----------
def view_database():
    db = load_db()
    if db.empty:
        return pd.DataFrame({"Message": ["Database is empty"]})
    return db

# ---------- 6. Gradio Interface ----------
with gr.Blocks(theme=gr.themes.Soft(), title="SmartMart QR Suite") as demo:
    gr.Markdown("""
    <h1 style="text-align:center;">🏪 SmartMart QR Suite</h1>
    <p style="text-align:center;">Generate product QR codes, store them in a central database, and scan in different store modes.</p>
    """)

    with gr.Tabs():
        # --- Generate Tab ---
        with gr.TabItem("📝 Generate QR"):
            with gr.Row():
                with gr.Column(scale=1):
                    prod_id = gr.Textbox(label="Product ID *", placeholder="e.g., PROD001")
                    prod_name = gr.Textbox(label="Product Name", placeholder="Organic Milk")
                    shelf = gr.Textbox(label="Shelf / Rack Number", placeholder="Rack-A12")
                    expiry = gr.Textbox(label="Expiry Date (YYYY-MM-DD)", placeholder="2026-06-15")
                    price = gr.Number(label="Selling Price ($)", value=0.0, precision=2)
                    status = gr.Radio(label="Status", choices=["not billed", "billed"], value="not billed")
                    gen_btn = gr.Button("Generate QR Code", variant="primary")
                with gr.Column(scale=1):
                    qr_image = gr.Image(label="QR Code Preview", type="pil", interactive=False)
                    gen_msg = gr.Textbox(label="Status", interactive=False)
                    qr_download = gr.File(label="📥 Download QR Image")

            gen_btn.click(
                fn=generate_qr,
                inputs=[prod_id, prod_name, shelf, expiry, price, status],
                outputs=[qr_image, gen_msg, qr_download]
            )

        # --- Scan Tab ---
        with gr.TabItem("📷 Scan QR"):
            with gr.Row():
                with gr.Column():
                    scan_file = gr.Image(label="Upload QR Image", type="filepath")
                    scan_mode = gr.Radio(
                        label="Scan Mode",
                        choices=["Shelf location", "POS (Billing)", "Gate (Exit check)"],
                        value="Shelf location"
                    )
                    scan_btn = gr.Button("Scan QR", variant="primary")
                with gr.Column():
                    scan_output = gr.Textbox(label="Scan Result", lines=5, interactive=False)

            scan_btn.click(
                fn=scan_qr,
                inputs=[scan_file, scan_mode],
                outputs=[scan_output]
            )

        # --- Database Tab ---
        with gr.TabItem("🗄️ Product Database"):
            refresh_btn = gr.Button("Refresh Database")
            db_display = gr.Dataframe(value=load_db(), interactive=False, label="Stored Products")
            refresh_btn.click(fn=view_database, outputs=[db_display])

    gr.Markdown("---\n*All QR data is stored in product_db.csv – persists across scans*")

# Launch
demo.launch(share=True, debug=False)

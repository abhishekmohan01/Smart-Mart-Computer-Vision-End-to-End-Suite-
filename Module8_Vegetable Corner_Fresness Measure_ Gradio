# ==============================================================
# SmartMart CV Suite – Module 8: Freshness Scoring + Rich Gradio Dashboard
# ==============================================================

# ---------- 0. Install everything ----------
!pip install -q kagglehub tensorflow matplotlib seaborn scikit-learn opencv-python pandas gradio

import kagglehub, os, cv2, numpy as np, pandas as pd, shutil, tempfile, io, base64
import matplotlib
matplotlib.use('Agg')   # for non‑interactive plotting in Colab
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from datetime import datetime, timedelta
import gradio as gr

print("✅ All libraries imported.")

# ---------- 1. Dataset download & preparation ----------
DATASET_DIR = "/content/fruits_dataset"
if not os.path.exists(os.path.join(DATASET_DIR, "train")):
    print("📥 Downloading dataset from Kaggle (13 MB)...")
    downloaded = kagglehub.dataset_download("nourabdoun/fruits-quality-fresh-vs-rotten")
    base = None
    for root, dirs, _ in os.walk(downloaded):
        if "train" in dirs and "valid" in dirs:
            base = root
            break
    if base is None:
        if "train" in os.listdir(downloaded) and "valid" in os.listdir(downloaded):
            base = downloaded
    if base is None:
        raise FileNotFoundError("Could not locate train/valid folders.")
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    shutil.copytree(base, DATASET_DIR)
    print("✅ Dataset extracted to", DATASET_DIR)
else:
    print("✅ Dataset already exists at", DATASET_DIR)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "valid")
print("Train classes:", os.listdir(train_dir))
print("Validation classes:", os.listdir(val_dir))

# ---------- 2. Data generators ----------
IMG_H, IMG_W = 224, 224
BATCH = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_H, IMG_W), batch_size=BATCH,
    class_mode='binary', shuffle=True
)
val_gen = valid_datagen.flow_from_directory(
    val_dir, target_size=(IMG_H, IMG_W), batch_size=BATCH,
    class_mode='binary', shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print("Class mapping:", train_gen.class_indices)

# ---------- 3. Build or load model ----------
MODEL_PATH = "/content/best_freshness.h5"
if not os.path.exists(MODEL_PATH):
    print("Building fresh model...")
    base_model = applications.MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(IMG_H, IMG_W, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)

    print("🚀 Training...")
    history = model.fit(train_gen, epochs=30, validation_data=val_gen,
                        callbacks=[early_stop, reduce_lr, checkpoint])
    print("✅ Training finished.")
else:
    print("⏳ Loading pre‑trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")

# ---------- 4. Freshness prediction (returns all UI elements) ----------
def predict_freshness_ui(files, expiry_string):
    """
    Returns: annotated_images, bar_chart_figure, html_report, csv_file_path
    """
    if not files:
        return [], None, "<p style='color:red;'>No images uploaded.</p>", None

    # Parse expiry dates
    expiry_dates = []
    if expiry_string.strip():
        dates = [d.strip() for d in expiry_string.split(";")]
    else:
        dates = []
    for i, _ in enumerate(files):
        if i < len(dates) and dates[i]:
            expiry_dates.append(dates[i])
        else:
            expiry_dates.append(None)

    # Determine fresh/rotten index
    if 'fresh' in class_names[0].lower():
        fresh_idx, rotten_idx = 0, 1
    else:
        fresh_idx, rotten_idx = 1, 0

    results = []
    annotated_images = []

    for idx, img_path in enumerate(files):
        # Preprocess
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_H, IMG_W))
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        prob = model.predict(arr, verbose=0)[0][0]
        if fresh_idx == 0:
            fresh_conf = (1 - prob) * 100
        else:
            fresh_conf = prob * 100
        rotten_conf = 100 - fresh_conf
        verdict = "FRESH" if fresh_conf >= 50 else "ROTTEN"

        # Annotate image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Color overlay
        overlay = img_rgb.copy()
        color = (0, 255, 0) if verdict == "FRESH" else (255, 0, 0)
        cv2.rectangle(overlay, (0, 0), (img_rgb.shape[1], 70), color, -1)
        alpha = 0.5
        img_rgb = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
        text = f"{verdict} | Conf: {fresh_conf:.1f}%"
        cv2.putText(img_rgb, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
        annotated_images.append(img_rgb)

        # Alert logic
        batch_expiry = expiry_dates[idx]
        alert_msg = ""
        days_left = "N/A"
        if batch_expiry:
            try:
                expiry_date = datetime.strptime(batch_expiry, '%Y-%m-%d')
                today = datetime.today()
                days_left = (expiry_date - today).days
                if days_left < 0:
                    alert_msg = "🚨 CRITICAL: Batch already expired! Remove immediately."
                elif verdict == "ROTTEN":
                    alert_msg = "⚠️ WARNING: Visually rotten, even though expiry not reached. Check storage."
                elif days_left <= 1:
                    alert_msg = "⏰ URGENT: Expires today/tomorrow. Move to front or discount."
                elif days_left <= 3:
                    alert_msg = f"📅 ALERT: Only {days_left} days left. Prioritize sale."
                elif fresh_conf < 70:
                    alert_msg = "🔍 NOTICE: Borderline freshness. Monitor closely."
                else:
                    alert_msg = f"✅ Fresh. {days_left} days until expiry."
                if verdict == "FRESH" and days_left < 0:
                    alert_msg += " (Product looks fresh but date expired – verify manually.)"
            except:
                alert_msg = "Invalid date format."
        else:
            alert_msg = "No expiry date provided."

        results.append({
            'File': os.path.basename(img_path),
            'Verdict': verdict,
            'Fresh_Conf': round(fresh_conf, 2),
            'Rotten_Conf': round(rotten_conf, 2),
            'Expiry_Date': batch_expiry if batch_expiry else 'N/A',
            'Days_Left': days_left if days_left != 'N/A' else 'N/A',
            'Alert': alert_msg
        })

    # ---- Create bar chart of freshness confidence ----
    df = pd.DataFrame(results)
    plt.figure(figsize=(8, len(df)*0.6 + 1.5))
    colors = ['#2ecc71' if v=='FRESH' else '#e74c3c' for v in df['Verdict']]
    bars = plt.barh(df['File'], df['Fresh_Conf'], color=colors, edgecolor='white', linewidth=1.5)
    plt.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Decision threshold (50%)')
    plt.xlabel('Freshness Confidence (%)', fontweight='bold')
    plt.xlim(0, 105)
    plt.legend(loc='lower right')
    # Add value labels
    for i, (val, conf) in enumerate(zip(df['Verdict'], df['Fresh_Conf'])):
        plt.text(conf+1.5, i, f"{conf:.1f}%", va='center', fontsize=11, fontweight='bold',
                 color='darkgreen' if val=='FRESH' else 'darkred')
    plt.tight_layout()
    # Convert figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    chart_img = base64.b64encode(buf.read()).decode('utf-8')

    # ---- HTML report with styled alerts ----
    html_rows = ""
    for _, row in df.iterrows():
        alert_class = "fresh" if row['Verdict'] == "FRESH" else "rotten"
        html_rows += f"""
        <div style="background: {'#e9f7ef' if row['Verdict']=='FRESH' else '#fdedec'}; 
                    padding: 10px; border-radius: 10px; margin: 10px 0;
                    border-left: 6px solid {'#2ecc71' if row['Verdict']=='FRESH' else '#e74c3c'};">
            <b style="font-size:1.1em;">📸 {row['File']}</b> — 
            <span style="color:{'#27ae60' if row['Verdict']=='FRESH' else '#c0392b'}; font-weight:bold;">
                {row['Verdict']}
            </span> (Fresh: {row['Fresh_Conf']}%, Rotten: {row['Rotten_Conf']}%)
            <br>
            <span style="color:#555;">📅 Expiry: {row['Expiry_Date']} &nbsp;|&nbsp; Days left: {row['Days_Left']}</span>
            <br>
            <span style="background:{'#f9e79f' if 'CRITICAL' in row['Alert'] or 'URGENT' in row['Alert'] else '#d6eaf8'}; 
                        padding: 3px 8px; border-radius: 5px;">
                {row['Alert']}
            </span>
        </div>
        """

    html_report = f"""
    <div style="font-family: Arial; max-width: 800px; margin: auto;">
        <h3 style="color:#2c3e50;">📋 Freshness Assessment Report</h3>
        {html_rows}
    </div>
    """

    # ---- Save CSV ----
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_csv.name, index=False)

    # The bar chart will be displayed using an HTML img tag (since we have base64)
    chart_html = f'<img src="data:image/png;base64,{chart_img}" style="width:100%; max-width:700px;">'

    return annotated_images, chart_html, html_report, tmp_csv.name

# ---------- 5. Build Gradio interface ----------
with gr.Blocks(theme=gr.themes.Soft(), title="SmartMart Freshness Scanner") as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>🍏🥑 SmartMart Freshness Scoring</h1>
        <p style="font-size:1.2em;">Upload fruit/vegetable images and (optionally) batch expiry dates.<br>
        AI predicts <b>freshness confidence</b>, <b>remaining shelf life</b>, and issues <b>actionable alerts</b>.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📤 Upload one or more images", file_count="multiple", type="filepath")
            expiry_input = gr.Textbox(
                label="🗓 Batch expiry dates (semicolon‑separated, one per image, e.g. 2026-05-10;2026-05-12)",
                placeholder="Leave blank to use visual assessment only"
            )
            analyze_btn = gr.Button("🔍 Analyze Freshness", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="🖼 Annotated Images", columns=2, height="auto", object_fit="contain")
    
    with gr.Row():
        chart_display = gr.HTML(label="📊 Freshness Confidence Chart")
    
    with gr.Row():
        report_display = gr.HTML(label="📋 Detailed Report with Alerts")

    with gr.Row():
        csv_download = gr.File(label="📥 Download CSV Report")

    analyze_btn.click(
        fn=predict_freshness_ui,
        inputs=[file_input, expiry_input],
        outputs=[gallery, chart_display, report_display, csv_download]
    )

    gr.Markdown("---\n*Powered by MobileNetV2 · Tiny dataset (13 MB) · Trained in Google Colab*")

# ---------- 6. Launch ----------
print("\n🌐 Launching Gradio interface...")
demo.launch(share=True, debug=False)

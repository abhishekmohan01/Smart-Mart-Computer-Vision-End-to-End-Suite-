# ==============================================================
# SmartMart CV Suite – Module 8: Complete Freshness Scoring
# (Fixed LR conflict, robust dataset handling)
# ==============================================================

# ---------- 0. Environment setup ----------
!pip install -q kagglehub tensorflow matplotlib seaborn scikit-learn opencv-python pandas

import kagglehub, os, cv2, numpy as np, pandas as pd, shutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from datetime import datetime, timedelta
from google.colab import files

print("✅ All libraries imported.")

# ---------- 1. Dataset download & preparation ----------
DATASET_DIR = "/content/fruits_dataset"
if not os.path.exists(os.path.join(DATASET_DIR, "train")):
    print("📥 Downloading dataset from Kaggle (13 MB)...")
    downloaded = kagglehub.dataset_download("nourabdoun/fruits-quality-fresh-vs-rotten")
    # kagglehub may return path to an archive or folder; we walk to find train/valid
    base = None
    for root, dirs, _ in os.walk(downloaded):
        if "train" in dirs and "valid" in dirs:
            base = root
            break
    if base is None:
        # Sometimes the downloaded folder itself is the base
        if "train" in os.listdir(downloaded) and "valid" in os.listdir(downloaded):
            base = downloaded
    if base is None:
        raise FileNotFoundError("Could not locate train/valid folders in the dataset.")
    # Copy to a simpler path (overwrite if exists)
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

# ---------- 2. Data generators with strong augmentation ----------
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
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH,
    class_mode='binary',      # fresh=0, rotten=1 (order depends on folder names)
    shuffle=True
)
val_gen = valid_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH,
    class_mode='binary',
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print("Class mapping (fresh vs rotten):", train_gen.class_indices)

# ---------- 3. Model building (MobileNetV2 transfer learning) ----------
base_model = applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_H, IMG_W, 3)
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

# ⚠️ FIX: Use a simple float learning rate (not a schedule) so ReduceLROnPlateau works
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# ---------- 4. Callbacks ----------
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = callbacks.ModelCheckpoint('best_freshness.h5', monitor='val_accuracy', save_best_only=True)

# ---------- 5. Training ----------
print("\n🚀 Training started...")
history = model.fit(
    train_gen,
    epochs=30,                # early stopping will cut it short
    validation_data=val_gen,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ---------- 6. Evaluation & visualizations ----------
# Plot training curves
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix & detailed metrics
val_gen.reset()
preds = model.predict(val_gen)
y_pred = (preds > 0.5).astype(int).flatten()
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ---------- 7. Freshness predictor function (enhanced) ----------
def predict_freshness(image_paths, expiry_dates=None):
    """
    image_paths: list of image file paths
    expiry_dates: optional list of expiry date strings (YYYY-MM-DD) or None per image
    Returns a list of dicts with predictions and alerts.
    """
    results = []
    # Determine which class index corresponds to fresh/rotten
    if 'fresh' in class_names[0].lower():
        fresh_idx = 0
        rotten_idx = 1
    else:
        fresh_idx = 1
        rotten_idx = 0

    for idx, img_path in enumerate(image_paths):
        # Load and preprocess
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_H, IMG_W))
        img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        prob = model.predict(img_arr, verbose=0)[0][0]
        if fresh_idx == 0:
            fresh_conf = (1 - prob) * 100
        else:
            fresh_conf = prob * 100
        rotten_conf = 100 - fresh_conf
        verdict = "FRESH" if fresh_conf >= 50 else "ROTTEN"

        # Display image with prediction
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4,4))
        plt.imshow(img_rgb)
        plt.title(f"{verdict} (fresh: {fresh_conf:.1f}% / rotten: {rotten_conf:.1f}%)", fontsize=12)
        plt.axis('off')
        plt.show()

        # Expiry date handling
        batch_expiry = expiry_dates[idx] if expiry_dates and idx < len(expiry_dates) else None
        alert_msg = ""
        days_left = None
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
                # Special case: fresh but expired
                if verdict == "FRESH" and days_left < 0:
                    alert_msg += " (Product looks fresh but date expired – verify manually.)"
            except:
                alert_msg = "Invalid date format."
        else:
            alert_msg = "No expiry date provided."

        # Store result
        results.append({
            'file': os.path.basename(img_path),
            'verdict': verdict,
            'fresh_confidence': round(fresh_conf, 2),
            'rotten_confidence': round(rotten_conf, 2),
            'expiry_date': batch_expiry if batch_expiry else 'N/A',
            'days_left': days_left if days_left is not None else 'N/A',
            'alert': alert_msg
        })

        # Print individual report
        print("-" * 50)
        print(f"📸 Image: {os.path.basename(img_path)}")
        print(f"   Verdict: {verdict} | Fresh conf: {fresh_conf:.2f}% | Rotten conf: {rotten_conf:.2f}%")
        print(f"   Alert: {alert_msg}")
        print("-" * 50)

    return results

# ---------- 8. Interactive upload & batch processing ----------
print("\n📤 UPLOAD FRUIT/VEGETABLE IMAGES (you can select multiple files)")
uploaded = files.upload()
if not uploaded:
    print("No files uploaded. Exiting.")
else:
    image_paths = list(uploaded.keys())
    print(f"📁 {len(image_paths)} image(s) received.")

    # Ask for expiry dates
    use_expiry = input("Do you want to enter batch expiry dates? (y/n): ").strip().lower()
    expiry_list = []
    if use_expiry == 'y':
        for img_name in image_paths:
            exp = input(f"Enter expiry date for '{img_name}' (YYYY-MM-DD, or press Enter to skip): ").strip()
            expiry_list.append(exp if exp else None)
    else:
        expiry_list = [None] * len(image_paths)

    # Run prediction
    print("\n🔍 Analyzing…")
    results = predict_freshness(image_paths, expiry_list)

    # ---------- 9. Summary table & CSV export ----------
    df = pd.DataFrame(results)
    print("\n📋 SUMMARY TABLE:")
    from IPython.display import display
    display(df)

    # Save to CSV
    csv_filename = 'freshness_report.csv'
    df.to_csv(csv_filename, index=False)
    print(f"✅ Report saved as {csv_filename} (download from Colab file panel).")

    # Optionally, show a summary bar chart of freshness confidence
    plt.figure(figsize=(8,4))
    colors = ['green' if v=='FRESH' else 'red' for v in df['verdict']]
    plt.barh(df['file'], df['fresh_confidence'], color=colors)
    plt.axvline(x=50, color='black', linestyle='--', label='Decision threshold (50%)')
    plt.xlabel('Freshness Confidence (%)')
    plt.title('Freshness Confidence per Image')
    plt.xlim(0,100)
    plt.legend()
    plt.tight_layout()
    plt.show()

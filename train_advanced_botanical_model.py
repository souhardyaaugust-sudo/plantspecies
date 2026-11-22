import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models

# -----------------------------------------------------------------------------
# Configuration & Dataset Directories
# -----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_SAMPLES_PER_CLASS = 45  # Balanced subset for fast, robust transfer learning
VAL_SPLIT = 0.2

PV_PATH = r"C:\Users\ASUS\.cache\kagglehub\datasets\abdallahalidev\plantvillage-dataset\versions\3\plantvillage dataset\color"
FLOWERS_PATH = r"C:\Users\ASUS\.cache\kagglehub\datasets\alxmamaev\flowers-recognition\versions\2\flowers"

print("--- COLLECTING DATASET IMAGES ---")
class_image_map = {}

# 1. PlantVillage Classes (38 classes)
if os.path.exists(PV_PATH):
    for c in sorted(os.listdir(PV_PATH)):
        c_dir = os.path.join(PV_PATH, c)
        if os.path.isdir(c_dir):
            imgs = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) >= 15:
                clean_name = c.replace("___", " - ").replace("_", " ")
                class_image_map[clean_name] = imgs[:MAX_SAMPLES_PER_CLASS]

# 2. Flower Classes (5 classes)
if os.path.exists(FLOWERS_PATH):
    for c in sorted(os.listdir(FLOWERS_PATH)):
        c_dir = os.path.join(FLOWERS_PATH, c)
        if os.path.isdir(c_dir):
            imgs = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            clean_name = f"Flower - {c.capitalize()}"
            class_image_map[clean_name] = imgs[:MAX_SAMPLES_PER_CLASS]

class_names = sorted(list(class_image_map.keys()))
num_classes = len(class_names)
print(f"Total Botanical Classes to Train: {num_classes}")

total_images = sum(len(v) for v in class_image_map.values())
print(f"Total Selected Training/Val Images: {total_images}")

# Save class label index
with open("botanical_classes.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# -----------------------------------------------------------------------------
# Feature Extraction with Pretrained MobileNetV2 Backbone
# -----------------------------------------------------------------------------
print("\n--- EXTRACTING DEEP FEATURES WITH MOBILENETV2 ---")
weights_path = os.path.abspath("weights/mobilenet_v2_weights_no_top.h5")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=weights_path, pooling="avg")

all_features = []
all_labels = []

t0 = time.time()
for label_idx, c_name in enumerate(class_names):
    img_paths = class_image_map[c_name]
    batch_imgs = []
    for p in img_paths:
        try:
            im = load_img(p, target_size=IMG_SIZE)
            arr = img_to_array(im)
            batch_imgs.append(arr)
        except Exception:
            continue
            
    if batch_imgs:
        batch_tensor = preprocess_input(np.array(batch_imgs, dtype=np.float32))
        feats = base_model(batch_tensor, training=False).numpy()
        all_features.append(feats)
        all_labels.extend([label_idx] * len(feats))

X = np.vstack(all_features)
y = np.array(all_labels)
print(f"Feature extraction complete in {time.time() - t0:.1f}s! Feature shape: {X.shape}")

# -----------------------------------------------------------------------------
# Train/Validation Split
# -----------------------------------------------------------------------------
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(len(X) * (1.0 - VAL_SPLIT))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# -----------------------------------------------------------------------------
# Build & Train Dedicated Classification Head
# -----------------------------------------------------------------------------
print("\n--- TRAINING CUSTOM BOTANICAL CLASSIFIER HEAD ---")
model = models.Sequential([
    layers.Input(shape=(1280,)),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=18,
    batch_size=32,
    verbose=1
)

val_acc = history.history["val_accuracy"][-1] * 100.0
print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")

# Save the trained model
model.save("custom_botanical_head.keras")
print("Saved custom trained botanical model to 'custom_botanical_head.keras'!")

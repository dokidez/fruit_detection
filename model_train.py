import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import gc
import joblib
import os

# Clear memory before starting
tf.keras.backend.clear_session()
gc.collect()

# Config
BATCH_SIZE = 8
IMG_SIZE = (128, 128)
EPOCHS = 8
DATASET_DIR = "Fruit_Train"

# FIX: Check dataset directory exists before loading
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"🚨 Dataset directory '{DATASET_DIR}' not found!")

print("Loading dataset for MobileNetV2...")
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=True, seed=42
)
NUM_CLASSES = len(dataset.class_names)

# IMPORTANT: Extract class names before splitting the dataset!
class_names = dataset.class_names
print(f"Found {NUM_CLASSES} classes: {class_names}")

# Split
dataset_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)

# FIX: Added .cache() and .shuffle() for training, .cache() for validation
# FIX: Use tf.data.AUTOTUNE instead of hardcoded 1
train_ds = (
    dataset
    .take(train_size)
    .cache()
    .shuffle(1000)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    dataset
    .skip(train_size)
    .take(val_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# Build Model
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)

# FIX: MobileNetV2 expects inputs in [-1, 1], but image_dataset_from_directory
# gives [0, 255]. This Rescaling layer fixes the mismatch.
# (Including it IN the model means your Streamlit app also auto-applies it)
x = layers.Rescaling(1./127.5, offset=-1)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
print("🚀 Training MobileNetV2...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

# Record Results
best_epoch = np.argmax(history.history['val_accuracy'])
mobilenet_train_acc = history.history['accuracy'][best_epoch]
mobilenet_val_acc = history.history['val_accuracy'][best_epoch]

print(f"\n✅ MobileNetV2 Results:")
print(f"Training Accuracy:   {mobilenet_train_acc:.4f}")
print(f"Validation Accuracy: {mobilenet_val_acc:.4f}")

# ==========================================
# SAVING THE MODELS FOR STREAMLIT
# ==========================================

# 1. Save in native Keras format (.keras) — RECOMMENDED
keras_model_path = "mobilenetv2_fruit_model.keras"
model.save(keras_model_path)
print(f"✅ Model saved in Keras format at: {keras_model_path}")

# 2. Save using Joblib / PKL (not recommended for Keras, but included)
pkl_model_path = "mobilenetv2_fruit_model.pkl"
try:
    joblib.dump(model, pkl_model_path)
    print(f"✅ Model saved in PKL/Joblib format at: {pkl_model_path}")
except Exception as e:
    print(f"⚠️ Could not save model as PKL. Error: {e}")

# 3. Save the class names — CRITICAL for Streamlit
class_names_path = "class_names.pkl"
joblib.dump(class_names, class_names_path)
print(f"✅ Class names saved at: {class_names_path}")
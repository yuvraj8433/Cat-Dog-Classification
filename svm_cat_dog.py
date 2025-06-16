import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# --- Constants ---
DATASET_DIR = "G:/Cat_vs_Dog/training_set"
IMG_SIZE = 128
CATEGORIES = ["cats", "dogs"]

# --- Load and preprocess data ---
def load_images(base_path, img_size):
    X, y = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(base_path, category)
        for file in tqdm(os.listdir(folder), desc=f"Loading {category}"):
            if file.lower().endswith(".jpg"):
                try:
                    img_path = os.path.join(folder, file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return np.array(X), np.array(y)

# Load dataset
print("📥 Loading data...")
X, y = load_images(DATASET_DIR, IMG_SIZE)
X = X / 255.0  # Normalize pixel values
y = to_categorical(y, num_classes=2)  # One-hot encoding
print(f"✅ Loaded {len(X)} images.")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training data for validation
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True
)
datagen.fit(X_train_new)

# Data generators
train_generator = datagen.flow(X_train_new, y_train_new, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

# --- CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Train the model ---
print("🚀 Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# --- Save model ---
model.save("cat_dog_cnn_model.keras")
print("✅ Model saved as cat_dog_cnn_model.keras")

# --- Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("cat_dog_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ TFLite model saved as cat_dog_cnn_model.tflite")

# --- Evaluate on Test Set ---
print("🔍 Evaluating model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("✅ Accuracy:", accuracy_score(y_true_classes, y_pred_classes))
print("\n📝 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=CATEGORIES))

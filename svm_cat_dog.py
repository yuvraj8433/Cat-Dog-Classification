import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_DIR = "G:/Cat_vs_Dog/training_set"
IMG_SIZE = 64

def load_images(base_path, img_size):
    X = []
    y = []
    categories = ["cats", "dogs"]
    for label, category in enumerate(categories):
        folder = os.path.join(base_path, category)
        for file in tqdm(os.listdir(folder), desc=f"Loading {category}"):
            if file.endswith(".jpg"):
                try:
                    img_path = os.path.join(folder, file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (img_size, img_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    X.append(img.flatten())
                    y.append(label)
                except:
                    print("Error loading:", img_path)
    return np.array(X), np.array(y)

# Load and train
print("Loading data...")
X, y = load_images(DATASET_DIR, IMG_SIZE)
print(f"Loaded {len(X)} images")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Save model
joblib.dump(svm, "svm_model.pkl")
print("Model saved as svm_model.pkl")

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

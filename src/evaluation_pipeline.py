import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import cv2

# ==============================
# CREATE OUTPUT FOLDER
# ==============================
os.makedirs("outputs_graphs", exist_ok=True)
# ==============================
# LOAD TRAINED CLASSIFIER
# ==============================
model = load_model("models/classifier.keras")

# ==============================
# DATASET PATH
# ==============================
data_dir = r"D:\GAN-IMG\Global-Challenge\Brain_Tumor_Dataset"

X = []
y_true = []

# ==============================
# LABEL MAPPING
# ==============================
labels = {"Negative": 0, "Positive": 1}

# ==============================
# LOAD IMAGES
# ==============================
for class_name in labels:
    folder = os.path.join(data_dir, class_name)

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)

        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        img = cv2.resize(img, (64, 64))

        # Normalize
        img = img / 255.0

        # Reshape
        img = np.reshape(img, (64, 64, 1))

        X.append(img)
        y_true.append(labels[class_name])

# Convert to numpy
X = np.array(X)
y_true = np.array(y_true)

# ==============================
# PREDICTION (TUNED)
# ==============================
preds = model.predict(X)

y_pred = []
for p in preds:
    if p < 0.98:   # threshold tuning
        y_pred.append(0)   # Normal
    else:
        y_pred.append(1)   # Tumor

y_pred = np.array(y_pred)

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Tumor"]
)

disp.plot(cmap='Blues', values_format='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.grid(False)

# ==============================
# SAVE GRAPH
# ==============================
plt.savefig("outputs_graphs/confusion_matrix.png", bbox_inches='tight')

# ==============================
# SHOW GRAPH
# ==============================
plt.show()
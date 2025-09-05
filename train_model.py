import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# STEP 3: Load annotations
df = pd.read_csv('_annotations.csv')
print("Total samples:", len(df))

# STEP 4: Prepare image and label data
IMG_SIZE = (224, 224)
X = []
y = []

for _, row in df.iterrows():
    img_path = os.path.join('train', row['filename'])
    if not os.path.exists(img_path):
        print(f"⚠️ File not found: {img_path}")
        continue
    image = load_img(img_path, target_size=IMG_SIZE)
    image = img_to_array(image) / 255.0
    X.append(image)
    y.append(row['class'])

X = np.array(X, dtype="float32")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("✅ Step 4 complete: Data prepared")

# STEP 5: Save label map
with open("labels.txt", "w") as f:
    for label in le.classes_:
        f.write(label + "\n")

print("✅ Step 5 complete: Labels saved")

# STEP 6: Build simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("✅ Step 6 complete: Model built")

# STEP 7: Train model
history = model.fit(X, y_cat, epochs=10, validation_split=0.2)

print("✅ Step 7 complete: Model trained")

# STEP 8: Export model for TFLite
print("⏳ Exporting model in SavedModel format...")
model.export("saved_model")   # ✅ correct way in TF 2.13 + Keras 2.13

# Convert to TFLite
print("⏳ Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ model.tflite and labels.txt saved!")

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training Accuracy")
plt.show()

print("🎉 All steps complete")

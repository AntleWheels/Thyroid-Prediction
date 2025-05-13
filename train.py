import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv("synthetic_thyroid_dataset.csv")
df.replace({'t': 1, 'f': 0}, inplace=True)
df['tumor'] = df['tumor'].astype(int)
df.dropna(inplace=True)
X = df.drop('tumor', axis=1)
y = df['tumor']
X = pd.get_dummies(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=17,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# Predict and evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Metrics report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ROC AUC score
auc = roc_auc_score(y_test, y_pred_prob)
print(f"🔵 ROC AUC Score: {auc:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

model.save("thyroid_detection_model.h5")
print("\n✅ Model saved as 'thyroid_prediction_model.h5'")
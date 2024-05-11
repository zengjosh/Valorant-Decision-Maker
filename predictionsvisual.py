import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the model
model = load_model('gun_shield_model.keras')

# Load and preprocess data
data = pd.read_csv('vct_data.csv')
encoder_shield = OneHotEncoder(sparse=False)
encoder_gun = OneHotEncoder(sparse=False)
encoder_shield.fit(data[['shield']])
encoder_gun.fit(data[['gun']])
scaler = StandardScaler()
features = ['u_credits', 't1_credits', 't2_credits', 't3_credits', 't4_credits']
X_scaled = scaler.fit_transform(data[features])
X_test = X_scaled
y_shield_test = encoder_shield.transform(data[['shield']])
y_gun_test = encoder_gun.transform(data[['gun']])

# Predictions and evaluations
y_gun_pred_probs = model.predict(X_test)[1]
y_gun_pred = np.argmax(y_gun_pred_probs, axis=1)
y_gun_test_labels = np.argmax(y_gun_test, axis=1)
gun_accuracy = accuracy_score(y_gun_test_labels, y_gun_pred)
print("Gun Accuracy:", gun_accuracy)
# print(classification_report(y_gun_test_labels, y_gun_pred, target_names=encoder_gun.categories_[0]))

y_shield_pred_probs = model.predict(X_test)[0]
y_shield_pred = np.argmax(y_shield_pred_probs, axis=1)
y_shield_test_labels = np.argmax(y_shield_test, axis=1)
shield_accuracy = accuracy_score(y_shield_test_labels, y_shield_pred)
print("Shield Accuracy:", shield_accuracy)

# Confusion Matrix for Shields
shield_cm = confusion_matrix(y_shield_test_labels, y_shield_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(shield_cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder_shield.categories_[0], yticklabels=encoder_shield.categories_[0])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Shield Classification')
plt.show()

# Confusion Matrix for Guns
gun_cm = confusion_matrix(y_gun_test_labels, y_gun_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(gun_cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder_gun.categories_[0], yticklabels=encoder_gun.categories_[0])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Gun Classification')
plt.show()

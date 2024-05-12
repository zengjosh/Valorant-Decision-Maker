import os
import cv2
import pytesseract
import numpy as np
import tensorflow as tf
from joblib import load
from tensorflow.keras.models import load_model

# Load the pre-trained model, scaler, and encoders
model = load_model('gun_shield_model.keras')
scaler = load('scaler.joblib')
encoder_shield = load('encoder_shield.joblib')
encoder_gun = load('encoder_gun.joblib')

def preprocess_image(image_path, region=None):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found or path is incorrect"
    h, w = img.shape[:2]
    if region is not None:
        x_pct, y_pct, w_pct, h_pct = region
        x = int(x_pct * w)
        y = int(y_pct * h)
        w = int(w_pct * w)
        h = int(h_pct * h)
        img = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(blur, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    text = pytesseract.image_to_string(eroded, config='--psm 6')
    return text

def process_credits(credits):
    # Strip unwanted characters and correct common OCR errors
    translations = {ord(c): None for c in ",o-O\n.nO "}
    credits = credits.translate(translations).lstrip('0')
    return '0' if credits == '' else credits[-4:]

def get_latest_image_path(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return max(all_files, key=os.path.getmtime) if all_files else None

# Use the function to get the latest image
directory_path = 'screenshots'
image_path = get_latest_image_path(directory_path)
if image_path:
    u_credits = int(process_credits(preprocess_image(image_path, region=(0.14322, 0.11713, 0.03932, 0.02546))))
    t1_credits = int(process_credits(preprocess_image(image_path, region=(0.15703, 0.25509, 0.02474, 0.01667))))
    t2_credits = int(process_credits(preprocess_image(image_path, region=(0.15703, 0.36011, 0.02474, 0.01667))))
    t3_credits = int(process_credits(preprocess_image(image_path, region=(0.15703, 0.46875, 0.02474, 0.01667))))
    t4_credits = int(process_credits(preprocess_image(image_path, region=(0.15703, 0.57439, 0.02474, 0.01667))))
    credits = np.array([[u_credits, t1_credits, t2_credits, t3_credits, t4_credits]])
    scaled_credits = scaler.transform(credits)
    predictions = model.predict(scaled_credits)
    predicted_shield = encoder_shield.inverse_transform(predictions[0])[0]
    predicted_gun = encoder_gun.inverse_transform(predictions[1])[0]
    print("Predicted Shield:", predicted_shield)
    print("Predicted Gun:", predicted_gun)
else:
    print("No images found in the specified directory.")

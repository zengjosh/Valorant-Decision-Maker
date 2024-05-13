import cv2
import pytesseract
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class_names = ['ares', 'bulldog', 'bulldog', 'classic', 'frenzy', 'ghost', 'guardian', 'judge', 'marshal', 'odin', 'operator', 'outlaw', 'phantom', 'sheriff', 'shorty', 'spectre', 'stinger', 'vandal']
shield_class_names = ['heavy', 'light', 'none']

def classify_shield(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    return shield_class_names[predicted_index]

def crop_image_path(image_path, region):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("The specified image does not exist.")
    if region:
        h, w = img.shape[:2]
        x = int(region[0] * w)
        y = int(region[1] * h)
        width = int(region[2] * w)
        height = int(region[3] * h)
        cropped_img = img[y:y + height, x:x + width]
        # print(x, y, x+width, y+height, image_path)
        return cropped_img
    return img

def process_credits(credits):
    credits = credits.replace(',', '')
    credits = credits.replace('§', '')
    credits = credits.replace('o', '')
    credits = credits.replace('-', '')
    credits = credits.replace('\n', '')
    credits = credits.replace('n', '')
    credits = credits.replace('.', '')
    credits = credits.replace('O', '')
    credits = credits.replace(' ', '')
    credits = credits.strip()
    if (len(credits) != 1):
        credits  =credits.lstrip('0')
    if len(credits) > 4:
        credits = credits[-4:]
    if (len(credits) == 0):
        credits = '0'
    return credits

def image_to_text(image_path, region=None):
    img = crop_image_path(image_path, region)

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
    # cv2.imshow('Final Preprocessed Image', eroded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(text)
    return text

def classify_gun(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    # plt.imshow(img)
    # plt.show()
    return predicted_class

def process_images(folder_path, model_path):
    data_records = []
    regions = [(0.44648, 0.71389, 0.04531, 0.02361), #u_amount spent
               (0.44531, 0.73819, 0.04648, 0.02222), #u_amount left
               (0.44531, 0.76458, 0.04727, 0.025), #t1_amount spent
               (0.44531, 0.79028, 0.04766, 0.02153), #t1_amount left
               (0.44531, 0.81667, 0.04766, 0.025), #t2_amount spent
               (0.44531, 0.84236, 0.04766, 0.02153), #t2_amount left
               (0.44531, 0.86667, 0.04766, 0.027), #t3_amount spent
               (0.44531, 0.89214, 0.04766, 0.02153), #t3_amount left
               (0.44531, 0.92083, 0.04766, 0.02561), #t4_amount spent
               (0.44531, 0.94514, 0.04766, 0.02003)] #t4_amount left
    gun_region = (0.37969, 0.71528, 0.06328, 0.04375)
    shield_region = (0.35195, 0.71250, 0.02461, 0.04931)

    counter = 2
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        credits = [image_to_text(image_path, region) for region in regions]
        cropped_gun_image = crop_image_path(image_path, gun_region)
        cropped_shield_image = crop_image_path(image_path, shield_region)

        create_temp_gun(cropped_gun_image, 'temp_gun', 'gun.png')
        create_temp_gun(cropped_shield_image, 'temp_shield', 'shield.png') 

        gun_name = classify_gun('vct_guns_model.keras', 'temp_gun/gun.png')
        shield_type = classify_shield('vct_shields_model.keras', 'temp_shield/shield.png')

        print(image_path)
        print(counter)
        counter += 1
        record = {
            'u_credits': (int(process_credits(credits[0]))+int(process_credits(credits[1]))),
            't1_credits': (int(process_credits(credits[2]))+int(process_credits(credits[3]))),
            't2_credits': (int(process_credits(credits[4]))+int(process_credits(credits[5]))),
            't3_credits': (int(process_credits(credits[6]))+int(process_credits(credits[7]))),
            't4_credits': (int(process_credits(credits[8]))+int(process_credits(credits[9]))),
            'shield': shield_type,
            'gun': gun_name
        }
        data_records.append(record)
    return data_records

def create_temp_gun(image, folder, name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Convert the numpy array image back to PIL Image for saving
    pil_image = Image.fromarray(image)

    # Save the image
    image_path = os.path.join(folder, name)
    pil_image.save(image_path)

def save_to_csv(records, output_file):
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)

folder_path = 'vctdata/'
model_path = 'vct_guns_model.h5'

#dataset based on the current top frag
records = process_images(folder_path, model_path)
save_to_csv(records, 'vct_data.csv')

print("Data processing complete, and CSV file is saved.")
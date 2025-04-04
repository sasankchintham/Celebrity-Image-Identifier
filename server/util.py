import cv2
import numpy as np
import joblib
import json
import os
import pywt

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CASCADE_PATH = os.path.join(MODEL_DIR, 'opencv', 'haarcascades', 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(MODEL_DIR, 'opencv', 'haarcascades', 'haarcascade_eye.xml')

# Globals
__model = None
__class_name_to_number = {}
__class_number_to_name = {}

def w2d(img, mode='haar', level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray) / 255

    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    img_reconstructed = pywt.waverec2(coeffs_H, mode)
    img_reconstructed = np.uint8(img_reconstructed * 255)

    return img_reconstructed

def get_cropped_image_if_2_eyes_from_array(img_bytes):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return img[y:y+h, x:x+w]

    return None

def classify_image(image_bytes):
    global __model
    if __model is None:
        load_saved_artifacts()

    cropped_img = get_cropped_image_if_2_eyes_from_array(image_bytes)
    if cropped_img is None:
        return {"error": "No face with 2 eyes detected."}

    img_resized = cv2.resize(cropped_img, (32, 32))
    img_har = w2d(img_resized, 'db1', 5)
    img_har_resized = cv2.resize(img_har, (32, 32))

    combined_img = np.vstack((img_resized.reshape(32*32*3, 1), img_har_resized.reshape(32*32, 1)))
    X_input = np.array(combined_img).reshape(1, -1)

    expected_feature_size = 4096
    if X_input.shape[1] != expected_feature_size:
        return {"error": f"Feature shape mismatch: Expected {expected_feature_size}, got {X_input.shape[1]}"}

    prediction = __model.predict(X_input)[0]

    if isinstance(prediction, int) and prediction in __class_number_to_name:
        return {"predicted_class": __class_number_to_name[prediction]}
    elif isinstance(prediction, str):
        return {"predicted_class": prediction}
    else:
        return {"error": f"Prediction '{prediction}' not found in class dictionary."}

def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name
    global __model

    print(" Loading saved artifacts...")
    with open(os.path.join(MODEL_DIR, 'class_dictionary.json'), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    __model = joblib.load(os.path.join(MODEL_DIR, "saved_model.pkl"))
    print(" Artifacts loaded successfully.")

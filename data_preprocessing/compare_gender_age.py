import os
import cv2
from deepface import DeepFace


dataset_directory = 'data/small/'

def get_gender_age_from_deepface(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    try:
        predictions = DeepFace.analyze(img, actions=['age', 'gender'], enforce_detection=False)
        first_prediction = predictions[0]
        age = int(first_prediction['age'])
        gender = 'm' if first_prediction['dominant_gender'] == 'Man' else 'f'
        return gender, age
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

total_checked = 0
mismatches = 0

for folder in os.listdir(dataset_directory):
    folder_path = os.path.join(dataset_directory, folder)
    if os.path.isdir(folder_path):
        folder_gender = folder.split('_')[-1]
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(folder_path, file)
                embedded_age = int(file.split('_')[-1].replace('.jpg', ''))

                predicted_gender, predicted_age = get_gender_age_from_deepface(image_path)

                if predicted_gender is not None and predicted_age is not None:
                    total_checked += 1
                    if predicted_gender != folder_gender or predicted_age != embedded_age:
                        mismatches += 1
                        print(f"Mismatch in {folder}/{file}: Expected {folder_gender}, {embedded_age}; Got {predicted_gender}, {predicted_age}")

print(f"Total checked: {total_checked}, Mismatches: {mismatches}")

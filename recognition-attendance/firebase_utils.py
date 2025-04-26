import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta

from firebase_configure import bucket, ref
from firebase_admin import storage


def load_known_faces_firebase(bucket_folder="known_faces"):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=bucket_folder + "/")

    images = []
    class_names = []

    for blob in blobs:
        if blob.name.endswith(('.png', '.jpg', '.jpeg')):
            img_bytes = blob.download_as_bytes()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                images.append(img)
                class_name = os.path.splitext(os.path.basename(blob.name))[0]
                class_names.append(class_name)

    print("Loaded known faces from Firebase:", class_names)
    return images, class_names

def load_unknown_faces_firebase(bucket_folder="unknown_faces"):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=bucket_folder + "/")

    encoded_unknowns = []
    unknown_names = []

    for blob in blobs:
        if blob.name.endswith(('.png', '.jpg', '.jpeg')):
            img_bytes = blob.download_as_bytes()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                if encodings:
                    encoded_unknowns.append(encodings[0])
                    name = os.path.splitext(os.path.basename(blob.name))[0]
                    unknown_names.append(name)

    print("Loaded unknown faces from Firebase:", unknown_names)
    return encoded_unknowns, unknown_names

def upload_images_to_firebase(folder_path, folder_name_in_firebase):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_path = os.path.join(folder_path, filename)
            blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename} to {folder_name_in_firebase}/ in Firebase.")
  
def upload_single_image_to_firebase(new_name, new_img, folder_name_in_firebase):
    try:
        # Encode the image 
        success, encoded_image = cv2.imencode('.jpg', new_img)
        if not success:
            print("Failed to encode image.")
            return

        # Upload to Firebase
        filename = f"{new_name}.jpg"
        blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
        blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')

        print(f"Uploaded {filename} directly to {folder_name_in_firebase}/ in Firebase.")

    except Exception as e:
        print(f"Error uploading image to Firebase: {e}")

def add_user_to_realtime_database(name, designation):
    data = {
        "name": name,
        "designation": designation,
        "total_attendance": 0,
        "last_attendance_time": None
    }
    ref.child(name).set(data)
    print(f"User {name} added to Firebase with designation {designation}.")

def remove_user_from_realtime_database(name):
    user_ref = ref.child(name)
    user_ref.delete()
    print(f"Removed {name} from Firebase Realtime Database.")

def remove_image_from_firebase(folder_name_in_firebase, filename):
    try:
        blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
        if blob.exists():
            blob.delete()
            print(f"Deleted {filename} from {folder_name_in_firebase}/ in Firebase.")
        else:
            print(f"{filename} does not exist in Firebase.")
    except Exception as e:
        print(f"Error deleting {filename} from Firebase: {e}")
    
def clear_unknown_faces_firebase(firebase_folder="unknown_faces"):
    try:
        for blob in bucket.list_blobs(prefix=f"{firebase_folder}/"):
            blob.delete()
            print(f"Deleted: {blob.name}")
    except Exception as e:
        print(f"Error clearing unknown faces from Firebase: {e}")

attendance_flags = {}

def update_attendance_firebase(name):
    now = datetime.now()

    # Check if user recognised in the last 10 seconds
    if name in attendance_flags:
        last_time = attendance_flags[name]
        if (now - last_time) < timedelta(seconds=10):
            print(f"{name}'s attendance marked just now")
            return

    # Update in Firebase
    user_ref = ref.child(name)
    user_data = user_ref.get()

    if user_data:
        new_total = user_data.get("total_attendance", 0) + 1
        user_ref.update({
            "total_attendance": new_total,
            "last_attendance_time": now.strftime("%Y-%m-%d %H:%M:%S")
        })

        attendance_flags[name] = now
        print(f"Attendance updated for {name}")

#  To manually add data to the database
# data = {
#     "Rohit": {
#         "name": "Rohit Karthick",
#         "role": "AI Intern",
#         "total_attendance": 5, 
#         "last_attendance_time": "2025-04-11 00:54:34"
#     },
#     "Dhanush": {
#         "name": "Dhanush",
#         "role": "Data Analyst",
#         "total_attendance": 2, 
#         "last_attendance_time": "2025-04-10 00:24:34"
#     },
# }

# for key, value in data.items():
#     ref.child(key).set(value)
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta
import uuid

from firebase_configure import bucket, ref
from firebase_admin import storage, db

def find_encodings(images):

    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encode_list.append(encodings[0])
        else:
            print("No face found in one of the images, skipping it.")

    return encode_list 
 
def load_faces_from_firebase(bucket_folder):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=f"{bucket_folder}/")

    images = []
    names = []

    for blob in blobs:
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_bytes = blob.download_as_bytes()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                images.append(img)
                name = os.path.splitext(os.path.basename(blob.name))[0]
                names.append(name)

    print(f"Loaded {len(images)} images from Firebase folder '{bucket_folder}': {names}")
    return images, names

def add_user_to_firebase(new_name, new_img, folder_name_in_firebase):
    try:
        success, encoded_image = cv2.imencode('.jpg', new_img)
        if not success:
            print("Failed to encode image.")
            return

        filename = f"{new_name}.jpg"
        blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
        blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')

        print(f"Added {filename} to {folder_name_in_firebase}/ in Firebase.")

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

def remove_user_from_firebase(user_name):    
    try:
        from firebase_utils import bucket
        blob = bucket.blob(f"known_faces/{user_name}.jpg")
        if blob.exists():
            blob.delete()
            print(f"Deleted {user_name}.jpg from Firebase Storage.")
        else:
            print(f"{user_name}.jpg does not exist in Firebase Storage.")
    except Exception as e:
        print(f"Error deleting {user_name}.jpg from Firebase: {e}")

def upload_unknown_face_to_firebase(img):
    try:
        success, encoded_image = cv2.imencode('.jpg', img)
        if not success:
            print("Failed to encode image.")
            return

        name = f"Unknown_{str(uuid.uuid4().hex[:4])}"
        time_now = datetime.now()
        filename = f"unknown_faces/{name}_{time_now}.jpg"

        blob = bucket.blob(filename)
        blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')

        db_ref = db.reference('Unknown Faces')
        db_ref.push({
            'id': name,
            'recognized_at': time_now.isoformat(),
            'image_url': f"gs://face-attendance-8a90a/{filename}"
        })

        print(f"Uploaded unknown face {name} to Firebase Storage and saved to Realtime Database")

    except Exception as e:
        print(f"Error uploading image to Firebase: {e}")

    return name

def clear_unknown_faces_firebase(firebase_folder="unknown_faces"):
    try:
        blobs = bucket.list_blobs(prefix=f"{firebase_folder}/")
        for blob in blobs:
            blob_name = blob.name
            blob.delete()
            print(f"Deleted: {blob_name}")
        
        db_ref = db.reference('Unknown Faces')
        db_ref.delete() 
        
        print("Cleared all records under 'Unknown Faces' from Realtime Database")
        
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
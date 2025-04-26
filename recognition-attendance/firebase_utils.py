import os
import cv2
from firebase_configure import bucket 
from firebase_configure import ref

def upload_images_to_firebase(folder_path, folder_name_in_firebase):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_path = os.path.join(folder_path, filename)
            blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename} to {folder_name_in_firebase}/ in Firebase.")
  
def upload_single_image_to_firebase(new_name, new_img, folder_name_in_firebase):
    try:
        # Encode the image to memory (as JPEG)
        success, encoded_image = cv2.imencode('.jpg', new_img)
        if not success:
            print("Failed to encode image.")
            return

        # Upload the encoded image directly to Firebase
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

def update_attendance_database():
    pass

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
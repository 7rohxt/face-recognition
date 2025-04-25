import os

from firebase_configure import bucket 
from firebase_configure import ref

data = {
    "Rohit": {
        "name": "Rohit Karthick",
        "role": "AI Intern",
        "total_attendance": 5, 
        "last_attendance_time": "2025-04-11 00:54:34"
    },
    "Dhanush": {
        "name": "Dhanush",
        "role": "Data Analyst",
        "total_attendance": 2, 
        "last_attendance_time": "2025-04-10 00:24:34"
    },
}

for key, value in data.items():
    ref.child(key).set(value)

def upload_images_to_firebase(folder_path, folder_name_in_firebase):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_path = os.path.join(folder_path, filename)
            blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename} to {folder_name_in_firebase}/ in Firebase.")

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
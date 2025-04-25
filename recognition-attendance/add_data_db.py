import firebase_admin
from firebase_admin import credentials, db, storage
import os

cred = credentials.Certificate("recognition-attendance/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-attendance-8a90a-default-rtdb.firebaseio.com/",
    'storageBucket': "face-attendance-8a90a"  
})

bucket = storage.bucket()

def upload_images_to_firebase(folder_path, folder_name_in_firebase):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_path = os.path.join(folder_path, filename)
            blob = bucket.blob(f"{folder_name_in_firebase}/{filename}")
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename} to {folder_name_in_firebase}/ in Firebase.")

ref = db.reference('Employees')

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

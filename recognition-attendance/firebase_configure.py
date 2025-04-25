import firebase_admin
from firebase_admin import credentials, db, storage
import os

cred = credentials.Certificate("recognition-attendance/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-attendance-8a90a-default-rtdb.firebaseio.com/",
    'storageBucket': "face-attendance-8a90a"  
})

bucket = storage.bucket()
ref = db.reference('Employees')
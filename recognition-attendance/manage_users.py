import cv2
import face_recognition
import os

from firebase_utils import upload_images_to_firebase
from firebase_utils import remove_image_from_firebase

def add_new_user(img):
    save_path = 'recognition-attendance/base-images'

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)

    if not encodings:
        print("No face found in the frame.")
        return None, None, None

    name = input("Enter name for the new face: ").strip()
    filename = f"{name}.jpg"
    full_path = os.path.join(save_path, filename)

    # Save image
    cv2.imwrite(full_path, img)
    print(f"Saved new face image as: {full_path}")

    upload_images_to_firebase(save_path, "known_faces")

    return encodings[0], name, img

def remove_user(user_name, encodings_list, class_names, image_list):
    if user_name in class_names:
        idx = class_names.index(user_name)
        
        file_dir = 'recognition-attendance/base-images'
        file_path = os.path.join(file_dir, f"{user_name}.jpg")
        
        # Remove from memory
        class_names.pop(idx)
        encodings_list.pop(idx)
        image_list.pop(idx)
        
        # Delete image file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Image '{file_path}' deleted.")
        else:
            print(f"Image file for user '{user_name}' not found.")

        

        print(f"User '{user_name}' removed successfully.")

    else:
        print(f"User '{user_name}' not found.")
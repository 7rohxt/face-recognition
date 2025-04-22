import cv2
import face_recognition
import os

def add_new_face(img):

    save_path = 'recognition-attendance/base-images'

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)

    if not encodings:
        print("No face found in the frame.")
        return None, None

    # Prompt user to enter name
    name = input("Enter name for the new face: ").strip()
    filename = f"{name}.jpg"
    full_path = os.path.join(save_path, filename)

    # Save image
    cv2.imwrite(full_path, img)
    print(f"Saved new face image as: {full_path}")

    return encodings[0], name
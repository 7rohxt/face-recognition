import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:  # Only append if a face is detected
            encodeList.append(encode[0])
        else:
            print("No face found in one of the images, skipping it.")
    return encodeList

def load_known_faces(path):
    images = []
    class_names = []
    my_list = os.listdir(path)
    print("Known faces:", my_list)

    for clss in my_list:
        current_img = cv2.imread(f'{path}/{clss}')
        images.append(current_img)
        class_names.append(os.path.splitext(clss)[0])  # splits .jpg
    print("Loaded known faces:", class_names)
    
    return images, class_names

def load_unknown_faces(unknown_path):
    encoded_unknowns = []
    unknown_names = []

    if os.path.exists(unknown_path):
        unknown_list = os.listdir(unknown_path)
        print("Unknowns found:", unknown_list)

        for file in unknown_list:
            img_path = os.path.join(unknown_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                # Append to main list
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                if encodings:
                    encoded_unknowns.append(encodings[0])
                    unknown_names.append(os.path.splitext(file)[0])
    
    return encoded_unknowns, unknown_names

def mark_attendance(name):
    file_path = 'attendance-log.csv'
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("Name,DateTime")
    
    with open(file_path, 'r+') as f:
        my_data_list = f.readlines()
        name_list = [line.split(',')[0] for line in my_data_list]
        if name not in name_list:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'\n{name},{dtString}')
            print(f"Wrote to CSV: {name},{dtString}")
        else:
            print(f"Already Marked for {name}")

def unknown_list(img):
    folder_path = 'unknowns'
    os.makedirs(folder_path, exist_ok=True)

    count = len(os.listdir(folder_path)) + 1
    name = f"Unknown{count}"
    filename = name + ".jpg"
    save_path = os.path.join(folder_path, filename)

    cv2.imwrite(save_path, img)
    print(f"Saved unknown face as {filename}")
    return name

# Mange Users

def add_new_user(img):
    save_path = 'base-images'

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

    return encodings[0], name, img

def remove_user(user_name, encodings_list, class_names, image_list):
    if user_name in class_names:
        idx = class_names.index(user_name)
        
        file_dir = 'base-images'
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

def clear_unknown_faces_local(local_dir="unknowns"):
 
    if os.path.exists(local_dir):
        for filename in os.listdir(local_dir):
            file_path = os.path.join(local_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted local unknown face: {filename}")
    else:
        print("Unknown faces folder not found locally.")

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from addface import add_new_face
 
path = 'recognition-attendance/base-images'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)

for clss in my_list:
    current_img = cv2.imread(f'{path}/{clss}')
    images.append(current_img)
    class_names.append(os.path.splitext(clss)[0]) # splits .jpg
print(class_names)


# Load unknowns (from previous sessions)
unknown_path = 'recognition-attendance/unknowns'
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
            images.append(img)
            class_names.append(os.path.splitext(file)[0])

            # Encode and store in unknown-specific list
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                encoded_unknowns.append(encodings[0])
                unknown_names.append(os.path.splitext(file)[0])

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0: # Only append if a face is detected
            encodeList.append(encode[0])
        else:
            print("No face found in one of the images, skipping it.")
    return encodeList
    
def mark_attendance(name):
    print(f"Marking attendance for {name}")  # Debug line
    file_path = 'recognition-attendance/attendance-log.csv'
    
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
            print("Already Marked")

def unknown_list(img):
    folder_path = 'recognition-attendance/unknowns'
    os.makedirs(folder_path, exist_ok=True)

    count = len(os.listdir(folder_path)) + 1
    name = f"Unknown{count}"
    filename = name + ".jpg"
    save_path = os.path.join(folder_path, filename)

    cv2.imwrite(save_path, img)
    print(f"Saved unknown face as {filename}")
    return name

encode_list_known = find_encodings(images)
print('Encoding Done')

# Display log in time in bounding box from attendance log 

# Load attendance log into a dictionary
attendance_log = {}
log_path = 'recognition-attendance/attendance-log.csv'

if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                name = parts[0].strip().upper()
                attendance_log[parts[0]] = parts[1]

# uploaded_image_path = 'recognition-attendance/test-images/Virat Kohli.jpg'  # Change path as needed
# img = cv2.imread(uploaded_image_path)

# for webcam 
cap = cv2.VideoCapture(0)

# frame_skip = 3
# frame_count = 0
while True:
    success, img = cap.read()
    # frame_count += 1

    # if frame_count % frame_skip != 0:
    #     continue

    scaled_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)

    current_face = face_recognition.face_locations(scaled_img)
    encodes_current_face = face_recognition.face_encodings(scaled_img, current_face)

    for encode_face, face_loc in zip(encodes_current_face, current_face):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_dis)

        if len(face_dis) > 0 and face_dis[match_index] < 0.50:
            name = class_names[match_index].upper()

        else:
            # Compare with previously saved unknowns
            unknown_matches = face_recognition.compare_faces(encoded_unknowns, encode_face)
            unknown_distances = face_recognition.face_distance(encoded_unknowns, encode_face)
            
            if len(unknown_distances) > 0 and min(unknown_distances) < 0.50:
                # If it matches an earlier unknown, reuse the name
                matched_index = np.argmin(unknown_distances)
                name = unknown_names[matched_index]
                print(f"Matched with a previous unknown: {name}")
            else:
                name = unknown_list(img)

                # Update the encoded_unknowns and unknown_names lists
                encoded = face_recognition.face_encodings(img)
                if encoded:
                    encoded_unknowns.append(encoded[0])
                    unknown_names.append(name)
                    
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        logged_time = attendance_log.get(name, "Time Not Found")

        # Display name
        cv2.rectangle(img, (x1, y2 - 55), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        # Display Logged In time
        cv2.putText(img, f"In Time: {logged_time}", (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
        
        mark_attendance(name)

    cv2.imshow('Webcam', img)
    # cv2.imshow('Uploaded Image', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        new_encoding, new_name = add_new_face(img)

        if new_encoding and new_name:
            encode_list_known.append(new_encoding)
            class_names.append(new_name)
            print(f"Added new face: {new_name}")

    elif key == ord('q'):
        break

    # cv2.destroyAllWindows()
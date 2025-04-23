import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from face_utils import find_encodings, mark_attendance, unknown_list, load_known_faces, load_unknown_faces
from manage_users import *
 
# Load known images
path = 'recognition-attendance/base-images'
images, class_names = load_known_faces(path)

# Load unknowns (from previous sessions)
unknown_path = 'recognition-attendance/unknowns'
encoded_unknowns, unknown_names = load_unknown_faces(unknown_path)

# Encode known faces
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
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if 'run_recognition' in locals() and run_recognition:
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if len(face_dis) > 0 and face_dis[match_index] < 0.50:
                name = class_names[match_index].upper()

            else:
                unknown_matches = face_recognition.compare_faces(encoded_unknowns, encode_face)
                unknown_distances = face_recognition.face_distance(encoded_unknowns, encode_face)

                if len(unknown_distances) > 0 and min(unknown_distances) < 0.50:
                    matched_index = np.argmin(unknown_distances)
                    name = unknown_names[matched_index]
                    print(f"Matched with a previous unknown: {name}")
                else:
                    name = unknown_list(img)

                    encoded = face_recognition.face_encodings(img)
                    if encoded:
                        encoded_unknowns.append(encoded[0])
                        unknown_names.append(name)

            logged_time = attendance_log.get(name, "Time Not Found")

            cv2.rectangle(img, (x1, y2 - 55), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"In Time: {logged_time}", (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

            mark_attendance(name)

    cv2.imshow('Webcam', img)
    # cv2.imshow('Uploaded Image', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        new_encoding, new_name = add_new_user(img)

        if new_encoding and new_name:
            encode_list_known.append(new_encoding)
            class_names.append(new_name)
            print(f"Added new face: {new_name}")

    elif key == ord('c'):
        run_recognition = True

    elif key == ord('q'):
        break

    # cv2.destroyAllWindows()
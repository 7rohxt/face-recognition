import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
from firebase_admin import storage

from face_utils import find_encodings, mark_attendance, unknown_list, load_known_faces, load_unknown_faces, clear_unknown_faces_local
from firebase_utils import add_user_to_realtime_database, remove_user_from_realtime_database
from firebase_utils import upload_images_to_firebase, remove_image_from_firebase, clear_unknown_faces_firebase, upload_single_image_to_firebase
from manage_users import add_new_user, remove_user

# Load known images
path = 'recognition-attendance/base-images'
images, class_names = load_known_faces(path)

# Load unknowns (from previous sessions)
unknown_path = 'recognition-attendance/unknowns'
encoded_unknowns, unknown_names = load_unknown_faces(unknown_path)

# Encode known faces
encode_list_known = find_encodings(images)
print('Encoding Done')

# Upload Images to firebase storage bucket
upload_images_to_firebase("recognition-attendance/base-images", "known_faces")
upload_images_to_firebase("recognition-attendance/unknowns", "unknown_faces")

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

# uploaded_image_path = 'recognition-attendance/test-images/Virat Kohli.jpg' 
# img = cv2.imread(uploaded_image_path)

# for webcam 
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    scaled_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)

    current_face = face_recognition.face_locations(scaled_img)
    encodes_current_face = face_recognition.face_encodings(scaled_img, current_face)

    for encode_face, face_loc in zip(encodes_current_face, current_face):
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw rectangle around the face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

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
                    upload_single_image_to_firebase(name, img, "unknown_faces")

        # Show name and attendance
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 12), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(img, f"In Time: {logged_time}", (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

        mark_attendance(name)

    # Continue displaying the webcam feed
    cv2.imshow('Webcam', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        new_encoding, new_name, new_image = add_new_user(img)

        if new_encoding is not None and new_name:
            print("Enter the designation")
            designation = input()
            upload_single_image_to_firebase(new_name, new_image, "known_faces")
            encode_list_known.append(new_encoding)
            class_names.append(new_name)
            images.append(new_image)
            print(f"Added new face: {new_name}")
            add_user_to_realtime_database(new_name,designation)
        
        # image_path = f"recognition-attendance/base-images/{new_name}.jpg"
        # upload_images_to_firebase(image_path, "images")

    elif key == ord('r'):
        # Remove user
        user_name = input("Enter the name of the user to remove: ").strip()
        remove_user(user_name, encode_list_known, class_names, images)
        remove_user_from_realtime_database(user_name)

        remove_image_from_firebase("known_faces", f"{user_name}.jpg")
    
    elif key == ord('u'):
        clear_unknown_faces_local()
        clear_unknown_faces_firebase()

    elif key == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()  
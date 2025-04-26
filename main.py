import cv2
import numpy as np
import face_recognition

from local_utils import (
    find_encodings, mark_attendance, unknown_list, 
    load_known_faces, load_unknown_faces, clear_unknown_faces_local,
    add_new_user, remove_user
)

from firebase_utils import (
    add_user_to_realtime_database, remove_user_from_realtime_database, update_attendance_firebase,
    clear_unknown_faces_firebase, upload_single_image_to_firebase, remove_user_from_firebase,
    load_known_faces_firebase, load_unknown_faces_firebase
)

USE_CLOUD = True

if USE_CLOUD:

    images, class_names = load_known_faces_firebase()
    encoded_unknowns, unknown_names = load_unknown_faces_firebase()
else:
    path = 'base-images'
    images, class_names = load_known_faces(path)

    unknown_path = 'unknowns'
    encoded_unknowns, unknown_names = load_unknown_faces(unknown_path)

encode_list_known = find_encodings(images)
print('Encoding Done')

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

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_dis)

        if len(face_dis) > 0 and face_dis[match_index] < 0.50:
            name = class_names[match_index]#.upper() ## mismatch in incrementing
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
                    if USE_CLOUD:
                        upload_single_image_to_firebase(name, img, "unknown_faces")

        # Display name in the bounding box
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 12), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        if USE_CLOUD:
            if name in class_names:
                update_attendance_firebase(name) 
        else:
            mark_attendance(name) ## update attendence in local excel sheet

    cv2.imshow('Webcam', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        new_encoding, new_name, new_image = add_new_user(img)

        if new_encoding is not None and new_name:
            print("Enter the designation")
            designation = input()

            if USE_CLOUD: 
                upload_single_image_to_firebase(new_name, new_image, "known_faces")
                add_user_to_realtime_database(new_name, designation) 
            else:  
                image_path = f"base-images/{new_name}.jpg"
                cv2.imwrite(image_path, new_image)

            encode_list_known.append(new_encoding)
            class_names.append(new_name)
            images.append(new_image)
            
            print(f"Added new face: {new_name}")

    elif key == ord('r'):
        user_name = input("Enter the name of the user to remove: ").strip()
       
        if USE_CLOUD:
            remove_user_from_realtime_database(user_name)  # Remove from Realtime Database separately
            remove_user_from_firebase(user_name, encode_list_known, class_names, images)

        else:

            remove_user(user_name, encode_list_known, class_names, images)
            
        print(f"User '{user_name}' removed successfully.")

    elif key == ord('u'):
        if USE_CLOUD:
            clear_unknown_faces_firebase()  
            print("Cleared unknown faces from Firebase.")
        else:
            clear_unknown_faces_local()  
        print("Cleared unknown faces from local storage.")


    elif key == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()  
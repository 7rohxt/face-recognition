import cv2
import numpy as np
import face_recognition

# Firebase utility functions 
from firebase_utils import (

    load_faces_from_firebase, find_encodings,

    add_user_to_storage, remove_user_from_storage,
    add_user_to_database, remove_user_from_database,
    
    load_attendance_flags, update_attendance_firebase,

    upload_unknown_face_to_firebase, 
    clear_unknown_faces_storage, clear_unknown_faces_database,

    execute_parallel_tasks
)

# Load all the users from firebase and encode the images
known_images, known_names = load_faces_from_firebase("known_faces")
encoded_knowns = find_encodings(known_images)

unknown_images, unknown_names = load_faces_from_firebase("unknown_faces")
encoded_unknowns = find_encodings(unknown_images)

print('Encoding Done')

# Load last attendance time for all users
load_attendance_flags()

# Initialize webcam
scaling_factor = 0.25
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    # Preprocess frame
    scaled_frame = cv2.resize(frame, (0, 0), fx=scaling_factor, fy=scaling_factor)
    scaled_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

    # Detect and encode faces
    current_face = face_recognition.face_locations(scaled_frame)
    encodes_current_face = face_recognition.face_encodings(scaled_frame, current_face)

    for encode_face, face_loc in zip(encodes_current_face, current_face):
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Match with known faces
        matches = face_recognition.compare_faces(encoded_knowns, encode_face)
        face_dis = face_recognition.face_distance(encoded_knowns, encode_face)
        match_index = np.argmin(face_dis)

        if len(face_dis) > 0 and face_dis[match_index] < 0.50:
            name = known_names[match_index]

        else:
            # Match with previously seen unknowns
            unknown_matches = face_recognition.compare_faces(encoded_unknowns, encode_face)
            unknown_distances = face_recognition.face_distance(encoded_unknowns, encode_face)

            if len(unknown_distances) > 0 and min(unknown_distances) < 0.50:
                matched_index = np.argmin(unknown_distances)
                name = unknown_names[matched_index]
                print(f"Matched with a previous unknown: {name}")
                
            else:
                # Upload new unknown face
                encodings = face_recognition.face_encodings(frame)
                new_name = upload_unknown_face_to_firebase(frame)

                if encodings:
                    encoded_unknowns.append(encodings[0])
                unknown_names.append(new_name)
                name = new_name

        # Display name on frame
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 12), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        # Update attendance
        if name in known_names:
            update_attendance_firebase(name)

    # Show webcam frame
    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1) & 0xFF


    # Add a new user
    if key == ord('n'):
        new_image = frame
        new_encoding_list = find_encodings([new_image]) ## function expects a list and returns a list

        if new_encoding_list is not None:
            new_encoding  = new_encoding_list[0]
            new_name = input("Enter name for the new face: ").strip()
            designation = input("Enter the designation")

            encoded_knowns.append(new_encoding)
            known_names.append(new_name)
            known_images.append(new_image)

            # Threads
            args1 = (new_name, new_image, "known_faces")
            args2 = (new_name, designation)

            execute_parallel_tasks(add_user_to_storage, 
                                   add_user_to_database, 
                                   args1, args2
                                   )
                                   

        print(f"Added new face: {new_name}")

    # Remove an existing user
    elif key == ord('r'):
        user_name = input("Enter the name of the user to remove: ").strip()

        if user_name in known_names:
            idx = known_names.index(user_name)
            known_names.pop(idx)
            encoded_knowns.pop(idx)
            known_images.pop(idx)

        # Threads
        execute_parallel_tasks(remove_user_from_storage, 
                               remove_user_from_database, 
                               user_name, user_name
                               )

        print(f"User '{user_name}' removed successfully.")

    # Clear all the unknows faces captured
    elif key == ord('u'):

        # Threads
        execute_parallel_tasks(clear_unknown_faces_storage, 
                               clear_unknown_faces_database,
                               "unknown_faces",
                               "Unknown Faces")

        print("Cleared unknown faces from Firebase.")
        
        encoded_unknowns.clear() 
        unknown_names.clear()

    # Exit the loop
    elif key == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()  
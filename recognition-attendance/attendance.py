import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
 
path = 'recognition-attendance/base-images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for clss in myList:
    current_img = cv2.imread(f'{path}/{clss}')
    images.append(current_img)
    classNames.append(os.path.splitext(clss)[0]) # splits .jpg
print(classNames)
 
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] # 0 takes first face only
        encodeList.append(encode)
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
 
encode_list_known = find_encodings(images)
print('Encoding Done')

uploaded_image_path = 'recognition-attendance/test-images/Virat Kohli.jpg'  # Change path as needed
img = cv2.imread(uploaded_image_path)

# for webcam 
# cap = cv2.VideoCapture(0)

# frame_skip = 3
# frame_count = 0
#  while True:
# success, img = cap.read()
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

    if face_dis[match_index] < 0.50:
        name = classNames[match_index].upper()
    else:
        name = 'Unknown'

    y1, x2, y2, x1 = face_loc
    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    mark_attendance(name)

# cv2.imshow('Webcam', img)
cv2.imshow('Uploaded Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
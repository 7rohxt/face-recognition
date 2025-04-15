import cv2
import face_recognition
import numpy
import dlib

img_base = face_recognition.load_image_file('recognition-attendance/images/Elon Musk.jpg')
img_base = cv2.cvtColor(img_base,cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('recognition-attendance/images/Bill Gates.jpg')
img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)
 
face_loc = face_recognition.face_locations(img_base)[0]
encode_elon = face_recognition.face_encodings(img_base)[0]
cv2.rectangle(img_base,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)
 
face_loc_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encode_elon],encode_test)
face_dis = face_recognition.face_distance([encode_elon],encode_test)
print(results,face_dis)
cv2.putText(img_test,f'{results} {round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Elon Musk',img_base)
cv2.imshow('Bill Gates',img_test)
cv2.waitKey(0)
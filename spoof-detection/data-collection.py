import cv2
from cvzone.FaceDetectionModule import FaceDetector

offsetPercentageW = 10
offsetPercentageH = 20




cap = cv2.VideoCapture(0)
detector = FaceDetector()
while True:
    success, img = cap.read()
    # imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        # bboxInfo - "id", "bbox", "score"."center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            print(x,y,w,h)



    cv2.imshow("Image", img)
    cv2.waitKey(1)
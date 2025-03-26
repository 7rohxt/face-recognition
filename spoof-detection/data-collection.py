from time import time

import cv2
from cvzone.FaceDetectionModule import FaceDetector
import cvzone

########################################################################
classID = 0 # 0 for fake and 1 for real
outputFolderPath = 'dataset/data-collect'
confidence = 0.75
save = True
blurThreshold = 35 # Largers is more focus

offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640,480
########################################################################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)


detector = FaceDetector()
while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw = False)

    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file

    if bboxs:
        # bboxInfo - "id", "bbox", "score"."center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = float(bbox["score"][0])
            print(x,y,w,h)

            # ------ Check the score ------
            if score > confidence:

            
                # ------ Adding an offset to the detected face ------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageW / 100) * h
                y = int(y - offsetH*4)
                h = int(h + offsetH*4)
                
                # ------ To avoid values below 0 ------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0


                # ------ Find Blurriness ------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ------ Normalization ------
                ih, iw, _ = img.shape
                xc, yc = x + w/2,  y + h/2
                
                xcn, ycn = round(xc / iw, 6), round(yc / ih, 6)
                wn, hn = round(w / iw, 6), round(h / ih, 6)

                # ------ To avoid values above 1 ------
                if xcn > 1: xcn = 0
                if ycn > 1: ycn = 0
                if wn > 1: wn = 0
                if hn > 1: hn = 0

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
                
                # ------ Drawing  ------
                cv2.rectangle(img,(x,y,w,h),(255,0,0),3)
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blurValue}', (x,y-20),
                                   scale = 1, thickness = 3)

        # ------ To save  ------
        if save:
            if all(listBlur) and listBlur != []:
                # ------  Save Image  --------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                # ------  Save Label Text File  --------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()
                
            


    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
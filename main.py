# 1. Setup

# 1.1 Importing Dependencies

import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

#import tensorflow dependencies - Functional API
#functional api are better than sequential apis when building hardcore dl models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# 1.2 GPU Setup
 
# Avoid out-of-memory erros by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus))

# 1.3 Setup directories

POSITIVE_PATH = os.path.join('data', 'positive')
NEGATIVE_PATH = os.path.join('data','negative')
ANCHOR_PATH = os.path.join('data','anchor')

os.makedirs(POSITIVE_PATH)
os.makedirs(NEGATIVE_PATH)
os.makedirs(ANCHOR_PATH)

# 2. Collecting Data Files

# 2.1 Collecting Negatives

# Uncompress Tar GZ
!tar -xf lfw.tgz

# Move LFW to the repo data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EXISTING_PATH = os.path.join('lfw', directory, file) 
        NEW_PATH = os.path.join(NEGATIVE_PATH, file)
        os.replace(EXISTING_PATH, NEW_PATH)

# 2.2 Collecting Anchors and Positives through OpenCv (webcam)

# Importing uuid Library generates unique names to images
import uuid

# Establishing a connection to the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Cut down frame to 250x250 pixels
    frame = frame[115:250+115, 195:250+195, :]

    # Collecting Anchors
    if cv2.waitKey(1) &0XFF == ord('a'):
        # Create the unique file path
        img_name = os.path.join(ANCHOR_PATH,'{}.jpg'.format(uuid.uuid1()))
        #Write out anchor image
        cv2.imwrite(img_name, frame)

        
    # Collecting positives
    if cv2.waitKey(1) &0XFF == ord('p'):
        # Create the unique file path
        img_name = os.path.join(POSITIVE_PATH,'{}.jpg'.format(uuid.uuid1()))
        #Write out anchor image
        cv2.imwrite(img_name, frame)

    # Display the image back to the screen
    cv2.imshow('Image Collection', frame)

    # Breaking Gracefully
    if cv2.waitKey(1) &0xff == ord('q'):
        break

# Release the webcam 
cap.release()
# Close the image show frame
cv2.destroyAllWindows()

# 3 


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

# 3 Load and Preprocess Images

# 3.1 Get Image Directories

anchor = tf.data.Dataset.list_files(ANCHOR_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POSITIVE_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEGATIVE_PATH+'\*.jpg').take(300)

ANCHOR_PATH+'\*.jpg' #looks for all jpg fils in the directory specified by anchorpath

# 3.2 Preprocessing - Scale and Resize

def preprocess(file_path):

    # Read in image from fil path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image o be 100x100x3
    img = tf.image.resize(img, (100,100))

    # Scaling image to be between 0 and 1
    img = img / 255.0

    return img

# 3.3 Create Labelled Dataset

# (anchor, positive) ==> 1,1,1,1,1
# (anchor, negative) ==> 0,0,0,0,0

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# 3.4 Build Train and Test portion

def preprocess_twin(input_img, validation_img, label): 
    return(preprocess(input_img), preprocess(validation_img), label)

# Build data loader pipeine
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Train Split
train_data = data.take(round(len(data)*.7)) # taking 70% of the data
train_data = train_data.batch(16)
train_data = train_data.prefetch(8) # This starts preprocessing next set to avoid bottleneck our neural network

# Test split
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# 4. Model Engineering

# 4.1 Embedding Layer

inp = Input(shape=(100,100,3), name='input_image')

c1 = Conv2D(64,(10,10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

c2 = Conv2D(128,(7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

c3 = Conv2D(128,(4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

c4 = Conv2D(1256,(4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(1024, activation = 'sigmoid')(f1)

mod = Model(inputs=[inp], outputs=[d1], name = 'embedding')

def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')

    # First block
    c1 = Conv2D(64,(10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128,(7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128,(4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    # Final block
    c4 = Conv2D(1256,(4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(1024, activation = 'sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name = 'embedding')

embedding = make_embedding()

# 4.2 Building Distance Layer

# Custom L1 distance layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
l1 = L1Dist()

# 4.3 Make Siamese Model

input_image = Input(name='input_img', shape= (100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)

siamese_layer = L1Dist()

distances = siamese_layer(inp_embedding, val_embedding)

classifier = Dense(1, activation = 'sigmoid')(distances)

siamese_network = Model(inputs=[input_image, validation_image], outputs = classifier, name ='SiameseNetwork')

def make_siamese_model():

    # Anchor image imput in the network
    input_image = Input(name='input_img', shape= (100,100,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Combine siamese distance calculation
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation = 'sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs = classifier, name ='SiameseNetwork')

siamese_model = make_siamese_model()

# 5. Training
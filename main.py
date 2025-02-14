# Importing Dependencies
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

# Avoid out-of-memory erros by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus))

#Setup paths
POSITIVE_PATH = os.path.join('data', 'positive')
NEGATIVE_PATH = os.path.join('data','negative')
ANCHOR_PATH = os.path.join('data','anchor')

os.makedirs(POSITIVE_PATH)
os.makedirs(NEGATIVE_PATH)
os.makedirs(ANCHOR_PATH)

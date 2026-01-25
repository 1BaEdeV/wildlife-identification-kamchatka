import numpy as np
from tensorflow import keras
from PIL import Image
import cv2

model = keras.models.load_model(r'E:\ULTIMATE_PROJECT\wildlife-identification-kamchatka\models\monkey_classifier.h5')
## import os
## import random
## from tqdm import tqdm
## import pandas as pd
## import matplotlib.pyplot as plt
## import seaborn as sns
## import cv2
## import random
## from IPython.display import Image
## import imutils

## from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## import numpy as np

## import tensorflow.keras as keras
## from tensorflow.keras.models import Sequential

## import tensorflow as tf
## from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator, array_to_img, img_to_array
## from tensorflow.keras.applications import EfficientNetB1
## from tensorflow.keras.models import Model
## from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, GlobalAveragePooling2D
## from tensorflow.keras.optimizers import Adam
## from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## import torch.optim as optim

## import torch
## import torch.nn as nn
import torchvision

## from common import get_data_from_json
## from kcn import ConvNeXtKAN
from common_testing import test_model

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"


if __name__ == '__main__':
    model = torchvision.models.efficientnet_b1(progress=True)

    test_model(model, "BasicMLP", num_epochs=30)

# TODO: do we need this?
    
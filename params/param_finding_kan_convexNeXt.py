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

import numpy as np

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

import torch, torchvision
import torch.nn as nn

## from common import get_data_from_json
from kcn import ConvNeXtKAN
from hyperopt import fmin, tpe, hp, Trials
from common_testing import test_model

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"


def create_model(params):
    model = ConvNeXtKAN(params)
    return model

space = {
    'grid_size': hp.choice('grid_size', np.arange(3, 15, dtype=int)),
    'spline_order': hp.choice('spline_order', np.arange(2, 5, dtype=int)),
    'scale_noise': hp.uniform('scale_noise', 0.1, 1.0),
    'scale_base': hp.uniform('scale_base', 0.1, 1.0),
    'scale_spline': hp.uniform('scale_spline', 0.1, 1.0)
}


def objective(params):
    i = params['grid_size']
    j = params['spline_order']
    k = params['scale_noise']
    l = params['scale_base']
    m = params['scale_spline']

    model_params = {
        'grid_size': i,
        'spline_order': j,
        'scale_noise': k,
        'scale_base': l,
        'scale_spline': m
    }

    model = create_model(model_params)
    # test_model vrací nějakou metriku, např. chybu nebo přesnost
    accuracy_testing, loss_validation = test_model(model, f"ConvNeXtKAN_{i}_{j}_{k}_{l}_{m}", num_epochs=1)

    return accuracy_testing[0]


if __name__ == '__main__':
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials)

    print("Best parameters:", best)

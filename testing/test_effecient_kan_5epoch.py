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

## import torch.optim as optim

import torch
import common_testing

## import torch.nn as nn
## import torchvision

## from common import get_data_from_json
## from kcn import ConvNeXtKAN
from common_testing import test_model
from efficientnet_kan import EfficientNetB1_KAN

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

common_testing.batch_size = 210

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(0)

    params = {
        'grid_size': 7,
        'spline_order': 3,
        'scale_noise': 0.97,
        'scale_base': 0.79,
        'scale_spline': 0.99
    }
    model = EfficientNetB1_KAN(4, params)
    test_model(model, f"Effective_b1_KAN_5epoch_7", num_epochs=30, progress_bar=False)

    params = {
        'grid_size': 64,
        'spline_order': 3,
        'scale_noise': 0.97,
        'scale_base': 0.79,
        'scale_spline': 0.99
    }
    model = EfficientNetB1_KAN(4, params)
    test_model(model, f"Effective_b1_KAN_5epoch_64", num_epochs=30, progress_bar=False)

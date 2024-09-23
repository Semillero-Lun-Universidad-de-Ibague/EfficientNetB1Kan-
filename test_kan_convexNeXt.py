import os
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from IPython.display import Image
import imutils

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np

import torch.optim as optim

import torch
import torch.nn as nn
import torchvision

from common import get_data_from_json
from kcn import ConvNeXtKAN
from common_testing import test_model

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"


def create_model(params):
    model = ConvNeXtKAN(params)
    return model


import common_testing

common_testing.batch_size = 1250

if __name__ == '__main__':
    grid_size = [32]
    spline_order = [3]
    scale_noise = [0.3]
    scale_base = [1.0]
    scale_spline = [0.2]

    for i in grid_size:
        for j in spline_order:
            for k in scale_noise:
                for l in scale_base:
                    for m in scale_spline:
                        params = {
                            'grid_size': i,
                            'spline_order': j,
                            'scale_noise': k,
                            'scale_base': l,
                            'scale_spline': m
                        }
                        model = create_model(params)
                        test_model(model, f"ConvNeXtKAN", num_epochs=50, progress_bar=False)
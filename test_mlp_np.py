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

import common_testing

common_testing.batch_size = 100

from common_testing import test_model

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

# batch_size = 300 # rtx5000 24gb :)

if __name__ == '__main__':
    model = torchvision.models.efficientnet_b1(progress=True, pretrained=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(0)
    test_model(model, "BasicMLP_np_50", num_epochs=50, progress_bar=False)

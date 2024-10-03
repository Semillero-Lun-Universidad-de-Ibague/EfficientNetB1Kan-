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

## import torch, torchvision
## import torch.nn as nn

## from common import get_data_from_json
## from kcn import ConvNeXtKAN
## from common_testing import test_model

## from efficientnet_kan import EfficientNetB1_KAN
## import numpy as np
## import nni
## from torch.optim.lr_scheduler import ReduceLROnPlateau
from nni.experiment import Experiment


search_space = {
    "grid_size": {
        "_type": "choice",
        "_value": [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            32,
            64
        ]
    },
    "spline_order": {
        "_type": "choice",
        "_value": [
            3,
            4,
            6
        ]
    },
    "scale_noise": {
        "_type": "uniform",
        "_value": [
            0.7,
            1.0
        ]
    },
    "scale_base": {
        "_type": "uniform",
        "_value": [
            0.7,
            1.0
        ]
    },
    "scale_spline": {
        "_type": "uniform",
        "_value": [
            0.7,
            1.0
        ]
    }
}

batch_size = 200

experiment = Experiment('local')
experiment.config.trial_command = 'python nni_test_effecient_kan.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.trial_concurrency = 1
experiment.config.max_experiment_duration = '80h'
experiment.run(8080)
experiment.stop()

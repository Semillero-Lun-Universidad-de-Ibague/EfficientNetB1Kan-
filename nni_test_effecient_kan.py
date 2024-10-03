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

import torch, nni
import common_testing
import torch.nn as nn
import torch.optim as optim
import numpy as np

from efficientnet_kan import EfficientNetB1_KAN
## from torch.optim.lr_scheduler import ReduceLROnPlateau
from nni.experiment import Experiment
## import torchvision

## from common import get_data_from_json
## from kcn import ConvNeXtKAN
from common_testing import prepare_dataset, update_json_with_key, train, validation, testing, Timer


experiment = Experiment('local')

device = torch.device("cuda" if torch.cuda.is_available() else "none")

common_testing.batch_size = 90


def main():
    # Get parameters from tuner
    prepare_dataset()

    params = nni.get_next_parameter()
    print(params)
    model = EfficientNetB1_KAN(num_classes=4, params=params)
    model = model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ## # Define the scheduler
    ## scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, verbose=True)

    ## # Initialize early stopping
    ## from common_testing import EarlyStopping

    criterion = nn.CrossEntropyLoss()

    accuracy, loss = (None, None)

    accuracy_validation, loss_validation = ([], [])

    time_per_epoch = Timer()
    time_trainings = []

    epochs = 5
    for epoch in range(epochs):
        time_per_epoch.start()
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(criterion, epoch, model, epochs, optimizer)
        accuracy, loss = validation(criterion, model)

        time_per_epoch.end()
        time_trainings.append(time_per_epoch.elapsed_time)

        tb_nni = {

            "time_for_epoch": time_per_epoch.elapsed_time,

            "default": accuracy,  # default metric
            "loss_testing": loss

        }

        nni.report_intermediate_result(tb_nni)

        accuracy_validation.append(accuracy)
        loss_validation.append(loss)

    accuracy_testing, loss_testing = testing(criterion, model)

    tb_nni = {
        "accuracy_validation": accuracy_validation,

        "loss_validation": loss_validation,

        "time_for_trainings": np.array(time_trainings).sum(),

        "default": accuracy_testing,  # default metric
        "loss_testing": loss_testing,
    }

    nni.report_final_result(tb_nni)

    tb_json = {
        "accuracy_validation": accuracy_validation,

        "loss_validation": loss_validation,

        "time_trainings": time_trainings,

        "model_name": f"EfficientNetB1_KAN_{params['grid_size']}_{params['spline_order']}_{params['scale_noise']}_{params['scale_base']}_{params['scale_spline']}",

        "num_epochs": epochs,

        "accuracy_testing": accuracy_testing,
        "loss_testing": loss_testing,
    }


    update_json_with_key("data.json", tb_json)


if __name__ == '__main__':
    main()

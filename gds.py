import cv2
import os
import numpy as np

_X_TRAIN = './train_folder/'
_X_TEST = './test_folder/'
_Y_TRAIN = 'train.csv'
_Y_TEST = 'test.csv'

def load_data():
    x_train = []
    for file in os.listdir(_X_TRAIN):
        x_train.append(cv2.imread(file))

    y_train = []
    with open(_Y_TRAIN, "r") as f:
        for line in f:
            y_train.append(line.split(", ")[1])

    x_test = []
    for file in os.listdir(_X_TEST):
        x_train.append(cv2.imread(file))

    y_test = []
    with open(_Y_TEST, "r") as f:
        for line in f:
            y_train.append(line.split(", ")[1])

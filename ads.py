import cv2
import os
import numpy as np

_X_TRAIN = './train_folder'
_X_TEST = './test_folder'
_Y_TRAIN = './y_train.csv'
_Y_TEST = './y_test.csv'


def load_data():
    x_train = []
    for file in os.listdir(_X_TRAIN):
        img = cv2.imread(os.path.join(_X_TRAIN, file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # mean = gray_img.mean(axis=0)
        # std = gray_img.mean(axis=0)
        # gray_img = (gray_img - mean) / std
        x_train.append(gray_img)

    y_train = []
    with open(_Y_TRAIN, "r") as f:
        for line in f:
            y_train.append(int(line.split(", ")[1]))

    x_test = []
    for file in os.listdir(_X_TEST):
        img = cv2.imread(os.path.join(_X_TEST, file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # mean = gray_img.mean(axis=0)
        # std = gray_img.mean(axis=0)
        # gray_img = (gray_img - mean) / std
        x_test.append(gray_img)

    y_test = []
    with open(_Y_TEST, "r") as f:
        for line in f:
            y_test.append(int(line.split(", ")[1]))

    # a = .astype("float32") / 255.0
    return (np.array(x_train).astype("float32") / 255.0, np.array(y_train),
            np.array(x_test).astype("float32") / 255.0, np.array(y_test))

import cv2
import math
import numpy as np

_TEST_IMAGE = "./t4.jpg"
_KERNEL = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])


class Stack:
    __layer_stack = None

    def add(self, layer):
        self.__layer_stack.append(layer)


class Conv2D:
    __filters = []
    __activation = None
    __feature_maps = None

    def __init__(self, kernel_num, kernel_size, activation, batch_normalization=True):
        for i in range(kernel_num):
            added_f = np.zeros(kernel_size)
            for h in range(kernel_size[0]):
                for w in range(kernel_size[1]):
                    added_f[h][w] = np.random.uniform(-2.0, 2.0)
            self.__filters.append(added_f)
        print(self.__filters)
        if activation.lower() == "relu":
            self.__activation = ReLu()

    def compute_feature_maps(self, input_d, shape):
        feature_maps = []
        for kernel in self.__filters:
            f_map = []
            for i in range(len(input_d)):
                if i == 0 or i == len(input_d) - 1:
                    continue
                row = []
                for j in range(len(input_d[i])):
                    if j == 0 or j == len(input_d[i]) - 1:
                        continue
                    val = (input_d[i-1][j-1] * kernel[0][0] +
                           input_d[i-1][j] * kernel[0][1] +
                           input_d[i-1][j+1] * kernel[0][2] +
                           input_d[i][j-1] * kernel[1][0] +
                           input_d[i][j] * kernel[1][1] +
                           input_d[i][j+1] * kernel[1][2] +
                           input_d[i+1][j-1] * kernel[2][0] +
                           input_d[i+1][j] * kernel[2][1] +
                           input_d[i+1][j+1] * kernel[2][2])
                    row.append(val)
                f_map.append(row)
            feature_maps.append(f_map)
        self.__feature_maps = np.array(feature_maps)
        return self.__activation.compute(self.__feature_maps)


class ReLu:
    def compute(self, input_f_maps):
        f_maps = []
        for f_map in input_f_maps:
            for i in range(len(f_map)):
                for j in range(len(f_map[i])):
                    f_map[i][j] = max(0, f_map[i][j])
            f_maps.append(f_map)
        return np.array(f_maps)


def convNet():

    img = cv2.imread(_TEST_IMAGE, 0)
    print(img)
    dst = preprocess_data(img)
    print(dst.shape)
    print(dst)
    model = Stack()
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convLayer(image, kernel):
    dst = []
    for i in range(len(image)):
        if i == 0 or i == len(image) - 1:
            continue
        row = []
        for j in range(len(image[i])):
            if j == 0 or j == len(image[i]) - 1:
                continue
            val = (image[i-1][j-1] * kernel[0][0] +
                   image[i-1][j] * kernel[0][1] +
                   image[i-1][j+1] * kernel[0][2] +
                   image[i][j-1] * kernel[1][0] +
                   image[i][j] * kernel[1][1] +
                   image[i][j+1] * kernel[1][2] +
                   image[i+1][j-1] * kernel[2][0] +
                   image[i+1][j] * kernel[2][1] +
                   image[i+1][j+1] * kernel[2][2]) / float(len(kernel))
            row.append(val)
        dst.append(row)
    dst = np.array(dst)
    dst = ReLu_layer(dst)
    return dst

def batch_norm_layer(input_data):
    dst = input_data.transpose()
    for i in range(len(dst)):

        emp_mean = 0.0
        for j in range(len(dst[i])):
            emp_mean += dst[i][j]
        emp_mean /= len(dst[i])

        variance = 0.0
        for j in range(len(dst[i])):
            variance += ((dst[i][j] - emp_mean) * (dst[i][j] - emp_mean))
        variance /= len(dst[i])

        for j in range(len(dst[i])):
            dst[i][i] = (input_data[i][j] - emp_mean) / math.sqrt(variance)

    dst = dst.transpose()
    return dst


def ReLu_layer(input_data):
    dst = np.arange(input_data.size, dtype=np.float).reshape(input_data.shape)
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            dst[i][j] = max(0, input_data[i][j])

    return dst


def preprocess_data(input_data):
    x_min = np.min(input_data)
    x_max = np.max(input_data)
    delta = float(x_max - x_min)
    # print(delta)
    dst = np.arange(input_data.size, dtype=np.float).reshape(input_data.shape)
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            dst[i][j] = (input_data[i][j] - x_min) / delta
    return dst 


def residual_block(image, kernel):
    l_branch = convLayer(image, kernel)
    l_branch = batch_norm_layer(l_branch)
    l_branch = convLayer(l_branch, kernel)
    l_branch = batch_norm_layer(l_branch)

    r_branch = convLayer(image, kernel)
    r_branch = batch_norm_layer(r_branch)

    for i in range(len(l_branch)):
        for j in range(len(l_branch[i])):
            l_branch[i][j] += r_branch[i][j]

    return l_branch

convNet()

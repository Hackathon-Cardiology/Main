import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import scipy.io
from scipy.signal import spectrogram
from scipy.signal import resample

def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try :
        tf.random.set_seed(seed)
    except :
        pass
    return
seed_everything(seed:=42)



def get_paths(seed=42) :
    """

    :return: lists of interictal and preictal paths to put in a dataset
    """

    types = ['interictal_segment', 'preictal_segment']
    interictal_paths = []
    preictal_paths = []

    for root, dirs, files in os.walk('./input'):
        for i, file in enumerate(files):
            path = os.path.join(root, file)
            segment = path[:-9]
            if segment.endswith(types[0]):
                interictal_paths.append(path)
                continue
            if segment.endswith(types[1]):
                preictal_paths.append(path)
            # there are test file types with no answer for a kaggle competition, so skip them

    # shuffle paths
    random.Random(seed).shuffle(interictal_paths)
    random.Random(seed).shuffle(preictal_paths)

    return interictal_paths,preictal_paths

interictal_paths,preictal_paths = get_paths(seed)

assert interictal_paths == get_paths(seed)[0]
assert preictal_paths == get_paths(seed)[1]


def load_from_paths(interictal_paths,preictal_paths,type_size=None) :
    """

    :param interictal_paths:
    :param preictal_paths:
    :param type_size:
    :return:
    """

    if type_size is None :
        type_size = len(interictal_paths)

    # for each file, preprocess the file and add to dataset
    assert type_size <= len(interictal_paths), f'Max type_size: {len(interictal_paths)}'
    len_preictal_paths = len(preictal_paths)

    for i in range(type_size) :
        path = interictal_paths[i]
        data = scipy.io.loadmat(path)
        filename = path.split('/')[-1]
        k = filename.split('_')[2]


        if i >= len_preictal_paths : continue

    return

load_from_paths(interictal_paths=interictal_paths,preictal_paths=preictal_paths,type_size=20)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

batch_size = 128
num_classes = 2
epochs = 100
img_rows, img_cols = 256,8

def build_model() :
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=None)
    ])

    return model
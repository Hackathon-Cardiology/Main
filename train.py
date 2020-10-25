import numpy as np
import tensorflow as tf
import random
import os
from scipy.signal import spectrogram
from scipy.signal import resample
import sys
sys.path.append("./")

from extract_transform_load import extractor

def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
            if not 'Patient_1' in file : continue
            if not file.endswith('.mat') : continue

            path = os.path.join(root, file)
            segment = path[:-9]

            if segment.endswith(types[0]):
                interictal_paths.append(path)
                continue
            if segment.endswith(types[1]):
                preictal_paths.append(path)
                continue
            # there are test file types with no answer for a kaggle competition, so skip them


    # shuffle paths
    random.Random(seed).shuffle(interictal_paths)
    random.Random(seed).shuffle(preictal_paths)

    return interictal_paths,preictal_paths

interictal_paths,preictal_paths = get_paths(seed)

assert interictal_paths == get_paths(seed)[0]
assert preictal_paths == get_paths(seed)[1]

def load_from_paths(interictal_paths, preictal_paths, num_samples=None, starting_point=0,seed=seed) :
    """

    :param interictal_paths: list of interictal data
    :param preictal_paths: list of preictal data
    :param num_samples:
    :param num_samples: where to start taking from the list
    :param starting_point:
    :param seed:
    :param seed: random state for shuffle
    :return: pandas dataframe of electrode values and whether they are preictal
    """

    X = []
    Y = []

    paths = interictal_paths + preictal_paths
    random.Random(seed).shuffle(paths)

    if num_samples is None :
        num_samples = len(paths)

    # for each file, preprocess the file and add to dataset
    if starting_point >= len(interictal_paths) : return
    if starting_point + num_samples > len(paths) : num_samples = len(paths) - starting_point

    for i in range(starting_point,num_samples + starting_point) :
        path = paths[i]
        x,y = extractor(path)


        d_array = x[12,:]

        secs = len(d_array) / 5000  # Number of seconds in signal X
        samps = secs * 500  # Number of samples to downsample
        dsample_array = resample(d_array, 300000)

        lst = list(range(300000))  # 3000000  datapoints initially
        for m in lst[::2000]:  # 5000 initial
            # Create a spectrogram every 2 second
            p_secs = dsample_array[m:m + 2000]  # d_array[0][m:m+15000]
            p_f, p_t, p_Sxx = spectrogram(p_secs, fs=500, return_onesided=False)
            p_SS = np.log1p(p_Sxx)
            arr = p_SS[:] / np.max(p_SS)
            X.append(arr)
            Y.append(y)

    return np.array(X),Y

secs = 3000000/5000 # Number of seconds in signal X
downsample = int(secs*500)     # Number of samples to downsample
X,y = load_from_paths(interictal_paths=interictal_paths, preictal_paths=preictal_paths, num_samples=None,
                     starting_point=0)

assert X.shape[0] == len(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling3D,GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

batch_size = 5
num_classes = 2
epochs = 100
img_rows, img_cols = 256,8

length = X.shape[0]
train_size = int(length * .8)
X_train = np.array(X[:train_size])
y_train = np.array(y[:train_size])
X_test = np.array(X[train_size:])
y_test = np.array(y[train_size:])

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def build_model() :
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3),
               activation='relu',
               input_shape=X_train[0].shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(.5),
        Dense(num_classes, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy',tf.keras.metrics.AUC()])
    return model

model = build_model()
model.fit(X_train,y_train,epochs=10,batch_size=batch_size)
model.evaluate(X_test,y_test)

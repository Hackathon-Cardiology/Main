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
seed_everything(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.append("./")

def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(seed:=42)

from etl import ETL

X,y = ETL('./input','Patient_1').extract_transform_load()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



length = X.shape[0]
train_size = int(length * .8)
X_train = np.array(X[:train_size])
y_train = np.array(y[:train_size])
X_test = np.array(X[train_size:])
y_test = np.array(y[train_size:])

num_classes = 1
batch_size = 5
epochs = 100

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

model.save('./model')

from sklearn.metrics import log_loss
print(log_loss(y_test,model.predict(X_test)))
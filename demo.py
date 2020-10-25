import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
import random
sys.path.append("./")
from etl import ETL

if not os.path.isdir('./input') :
    os.mkdir('./input')


def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(seed:=42)

X,y = ETL('./input','Patient_1').extract_transform_load()

length = X.shape[0]
train_size = int(length * .8)
train_data = list(zip(X,y))
random.Random(seed).shuffle(train_data)
X,y = zip(*train_data)
X_train = np.array(X[:train_size])
y_train = np.array(y[:train_size])
X_test = np.array(X[train_size:])
y_test = np.array(y[train_size:])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
def build_model() :
    num_classes = 1
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3),
               activation='relu',
               input_shape=X_train[0].shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(.4),
        Dense(num_classes, activation='sigmoid')
    ])



    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy',tf.keras.metrics.AUC()])
    return model

model = build_model()
model.fit(X_train,y_train,epochs=15,batch_size=None)
model.evaluate(X_test,y_test)

predictions = np.array(model.predict(X_test)).astype(np.float64)

from sklearn.metrics import log_loss,accuracy_score,roc_auc_score,plot_roc_curve,confusion_matrix

print('\nlog loss:')
print(log_loss(y_true=y_test,y_pred=predictions,labels=[0,1]))

try :
    print('\nroc:')
    print(roc_auc_score(y_true=y_test, y_score=predictions,labels=[0,1]))
except ValueError :
    print(np.nan)

print('\naccuracy:')
acc_preds = predictions.copy()
for i in range(len(predictions)) :
    if predictions[i] > .5 :
        acc_preds[i] = 1
    else : acc_preds[i] = 0
print(accuracy_score(y_true=y_test,y_pred=acc_preds))

print(f'\nConfusion Matrix :\n{confusion_matrix(y_test,acc_preds)}')
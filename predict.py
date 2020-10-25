import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("./")
from etl import ETL

X,y = ETL('./input','Patient_2').extract_transform_load()
model = tf.keras.models.load_model('./model')

predictions = np.array(model.predict(X)).astype(np.float64)

from sklearn.metrics import log_loss,accuracy_score,roc_auc_score

print('log loss')
print(log_loss(y_true=y,y_pred=predictions))


print('roc')
print(roc_auc_score(y_true=y, y_score=predictions))


print('accuracy')
for i in range(len(predictions)) :
    if predictions[i] > .5 :
        predictions[i] = 1
    else : predictions[i] = 0
print(accuracy_score(y_true=y,y_pred=predictions))


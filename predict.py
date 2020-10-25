import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("./")
from etl import ETL

X,y = ETL('./input',['Patient_1_preictal_segment_0005.mat']).extract_transform_load()
model = tf.keras.models.load_model('./model')
print(X.shape)
predictions = np.array(model.predict(X)).astype(np.float64)

from sklearn.metrics import log_loss,accuracy_score,roc_auc_score,plot_roc_curve,confusion_matrix

print('\nlog loss:')
print(log_loss(y_true=y,y_pred=predictions,labels=[0,1]))

try :
    print('\nroc:')
    print(roc_auc_score(y_true=y, y_score=predictions,labels=[0,1]))
except ValueError :
    print(np.nan)
print('\naccuracy:')
acc_preds = predictions.copy()
for i in range(len(predictions)) :
    if predictions[i] > .5 :
        acc_preds[i] = 1
    else : acc_preds[i] = 0
print(accuracy_score(y_true=y,y_pred=acc_preds))

print(confusion_matrix())
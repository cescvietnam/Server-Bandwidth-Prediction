import numpy as np
import pandas as pd
import keras
import os.path
import time

from keras.utils import multi_gpu_model
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD

start = time.time()
print(time)
test = pd.read_csv("Data/scaled_test_data.csv")
n_test_samples = len(test.index)

features = ["HOUR_ID"]
features.extend(test.columns.tolist()[6:])

print(time.time())
print(features)

inputs = Input(shape=(len(features),),name='input')
x = Dense(1000, activation='relu')(inputs)
#x = Dropout(0.3)(x)
x = Dense(500, activation='relu')(x)
#x = Dropout(0.3)(x)
x = Dense(200, activation='relu')(x)
#x = Dropout(0.2)(x)
x1 = Dense(100, activation='relu')(x)
x2 = Dense(100, activation='relu')(x)
#x1 = Dropout(0.2)(x1)
#x2 = Dropout(0.2)(x2)
x1 = Dense(50, activation='relu')(x1)
x2 = Dense(50, activation='relu')(x2)
x1 = Dense(20, activation='relu')(x1)
x2 = Dense(20, activation='relu')(x2)
out1 = Dense(1, activation='relu', name='out1')(x1)
out2 = Dense(1, activation='relu', name='out2')(x2)
model = Model(inputs=inputs, outputs=[out1, out2])

parallel_model = multi_gpu_model(model, gpus=2)

parallel_model.load_weights("Model/best_direct_training_fc_model.hdf5")

#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

parallel_model.compile(loss='mape', loss_weights=[0.8, 0.2], optimizer=optimizer, metrics=['mape'])
print("Model compiled successfully")

predictions = parallel_model.predict(test[features])

submission = pd.DataFrame(columns=["id","label"])
submission["id"] = test["id"]
submission["label"] = ["{:.2f}".format(predictions[0][i][0]) + " " + "{:.0f}".format(round(predictions[1][i][0])) for i in test.index]
submission.to_csv("Submission/submission.csv", index=False)

print(predictions[:10])

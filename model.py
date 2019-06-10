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
data = pd.read_csv("Data/scaled_train_data.csv")
n_train_samples = len(data.index)
print(n_train_samples)
features = ["HOUR_ID"]
features.extend(data.columns.tolist()[7:])
targets = data.columns.tolist()[5:7]
batch_size=1024
n_val_data = 100000
val_data = data.iloc[-n_val_data:,:]
validation_split = 0.05
print(time.time())
print(features)
print(targets)

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
if os.path.isfile("Model/best_direct_training_fc_model.hdf5"):
	parallel_model.load_weights("Model/best_direct_training_fc_model.hdf5")
	print("Model loaded successfully")

#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
optimizer = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

parallel_model.compile(loss='mape', loss_weights=[0.8, 0.2], optimizer=optimizer, metrics=['mape'])
print("Model compiled successfully")
filepath = "Model/best_direct_training_fc_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print("Training ready")

parallel_model.fit(data[features], {'out1':data[targets[0]], 'out2':data[targets[1]]}, batch_size=batch_size, validation_split=validation_split, epochs=100, shuffle=True, callbacks=callbacks_list, verbose=1)


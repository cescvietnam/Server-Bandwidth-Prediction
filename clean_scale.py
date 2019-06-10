import numpy as np
import pandas as pd
import os.path
import time

from sklearn.preprocessing import MinMaxScaler


start = time.time()
print(time)
data = pd.read_csv("Data/preprocessed_train_data.csv")
print(len(data.index))

features = ["HOUR_ID", "YEAR", "MONTH", "DATE"]

print(time.time())
print(features)

data.drop(data[(data.BANDWIDTH_TOTAL<0.000001)|(data.BANDWIDTH_TOTAL>1000000)|(data.MAX_USER<1)|(data.MAX_USER>1000000)].index, inplace=True)
n_train_samples = len(data.index)
print(n_train_samples)

scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(data[features])
data[features] = scaler.transform(data[features])
data.to_csv("Data/scaled_train_data.csv", index=False)
print(data.columns)
print(data.head())

data = pd.read_csv("Data/preprocessed_test_data.csv")
data[features] = scaler.transform(data[features])
data.to_csv("Data/scaled_test_data.csv", index=False)
print(data.columns)
print(data.head())

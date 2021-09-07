# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:49:00 2021

@author: Kushal
"""

import pandas as pd
import datetime as dt
import numpy as np
import os
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import math
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



data = pd.read_excel('./GEFCom2014 Data/GEFCom2014-E_V2/GEFCom2014-E.xlsx',sheet_name='Hourly')
data = data.dropna()
data.Date +=  pd.to_timedelta(data.Hour, unit='h')

data = data.set_index('Date')
data = data.shift(-1)
data = data['2012':'2014']
ts_data = data.copy()
ts_data = ts_data[['load','T']]
ts_data_load = ts_data['load']


ts_data.dtypes

ts_data.describe()

decomposition = sm.tsa.seasonal_decompose(ts_data_load, model = 'additive')
fig = decomposition.plot()


valid_st_data_load = "2014-09-01 00:00:00"
test_st_data_load = "2014-11-01 00:00:00"


T = 6
HORIZON = 1
train = pd.DataFrame(ts_data_load.copy()[ts_data_load.index < valid_st_data_load])[["load"]]

scaler = MinMaxScaler()
train["load"] = scaler.fit_transform(train)

train_shifted = train.copy()
train_shifted["y_t+1"] = train_shifted["load"].shift(-1, freq="H")
for t in range(1, T + 1):
    train_shifted[str(T - t)] = train_shifted["load"].shift(T - t, freq="H")
y_col = "y_t+1"
X_cols = ["load_t-5", "load_t-4", "load_t-3", "load_t-2", "load_t-1", "load_t"]
train_shifted.columns = ["load_original"] + [y_col] + X_cols

train_shifted = train_shifted.dropna(how="any")
train_shifted.head(5)


# Step: transform this pandas dataframe into a numpy array
y_train = train_shifted[y_col].to_numpy()
X_train = train_shifted[X_cols].to_numpy()

X_train = X_train.reshape(X_train.shape[0], T, 1)
y_train.shape
y_train[:3]
X_train.shape
X_train[:3]

train_shifted.head(3)

look_back_dt = dt.datetime.strptime(valid_st_data_load, "%Y-%m-%d %H:%M:%S") - dt.timedelta(
    hours=T - 1
)
valid = pd.DataFrame(ts_data_load.copy()[(ts_data_load.index >= look_back_dt) & (ts_data_load.index < test_st_data_load)])
[["load"]]

valid["load"] = scaler.transform(valid)

valid_shifted = valid.copy()
valid_shifted["y+1"] = valid_shifted["load"].shift(-1, freq="H")
for t in range(1, T + 1):
    valid_shifted["load_t-" + str(T - t)] = valid_shifted["load"].shift(T - t, freq="H")

valid_shifted = valid_shifted.dropna(how="any")
valid_shifted.head(3)

y_valid = valid_shifted['y+1'].to_numpy()
X_valid = valid_shifted[["load_t-" + str(T - t) for t in range(1, T + 1)]].to_numpy()
X_valid = X_valid.reshape(X_valid.shape[0], T, 1)

y_valid.shape
y_valid[:3]
X_valid.shape
X_valid[:3]


# RUNNING THE GRU

LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 10


model = Sequential()
model.add(GRU(LATENT_DIM, input_shape=(T, 1)))
model.add(Dense(HORIZON))
model.compile(optimizer="RMSprop", loss="mse")
model.summary()



earlystop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

model_history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_valid, y_valid),
    callbacks=[earlystop],
    verbose=1,
)



######################################
# TEST SET

look_back_dt = dt.datetime.strptime(test_st_data_load, "%Y-%m-%d %H:%M:%S") - dt.timedelta(
    hours=T - 1
)
test = pd.DataFrame(ts_data_load.copy()[test_st_data_load:])

test["load"] = scaler.transform(test)

test_shifted = test.copy()
test_shifted["y_t+1"] = test_shifted["load"].shift(-1, freq="H")
for t in range(1, T + 1):
    test_shifted["load_t-" + str(T - t)] = test_shifted["load"].shift(T - t, freq="H")

test_shifted = test_shifted.dropna(how="any")

y_test = test_shifted["y_t+1"].to_numpy()
X_test = test_shifted[["load_t-" + str(T - t) for t in range(1, T + 1)]].to_numpy()
X_test = X_test.reshape(X_test.shape[0], T, 1)

y_test.shape
X_test.shape



ts_predictions = model.predict(X_test)
ts_predictions

ev_ts_data = pd.DataFrame(
    ts_predictions, columns=["t+" + str(t) for t in range(1, HORIZON + 1)]
)
ev_ts_data["timestamp"] = test_shifted.index
ev_ts_data = pd.melt(ev_ts_data, id_vars="timestamp", value_name="prediction", var_name="h")
ev_ts_data["actual"] = np.transpose(y_test).ravel()
ev_ts_data[["prediction", "actual"]] = scaler.inverse_transform(
    ev_ts_data[["prediction", "actual"]]
)
ev_ts_data.head()

def mape(ts_predictions, actuals):
    """Mean absolute percentage error"""
    return ((ts_predictions - actuals).abs() / actuals).mean()

mape(ev_ts_data["prediction"], ev_ts_data["actual"])

ev_ts_data[ev_ts_data.timestamp < "2014-11-08"].plot(
    x="timestamp", y=["prediction", "actual"], style=["r", "b"], figsize=(15, 8)
)
plt.xlabel("timestamp", fontsize=12)
plt.ylabel("load", fontsize=12)
plt.show()



































































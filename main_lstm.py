# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:21:00 2021

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
import seaborn as sns
from statsmodels.tsa.stattools import adfuller




data = pd.read_excel('./GEFCom2014 Data/GEFCom2014-E_V2/GEFCom2014-E.xlsx',sheet_name='Hourly')
data = data.dropna()
data.Date +=  pd.to_timedelta(data.Hour, unit='h')
data = data.set_index('Date')
ts_data = data.copy()
ts_data = ts_data[['load','T']]
ts_data_load = ts_data['load']
ts_data.dtypes
ts_data.describe()

decomposition = sm.tsa.seasonal_decompose(ts_data_load['2012-07-01':'2012-12-31'], model = 'additive')
fig = decomposition.plot()



#********************************************************

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()
    
    plt.figure(figsize=(14,5))
    sns.despine(left=True)
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    
test_stationarity(ts_data_load.dropna())



####################################################


dataset = ts_data['load'].values #numpy.ndarray
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

valid_st_data_load = '2012-01-01 00:00:00'
test_st_data_load = '2014-01-01 00:00:00'

# train
train = pd.DataFrame(ts_data_load.copy()[ts_data_load.index < test_st_data_load])#[['load']]
train = train['load'].values
train = train.astype('float32')
train = np.reshape(train, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)

# valid
valid = pd.DataFrame(ts_data_load.copy().loc[(ts_data_load.index >= valid_st_data_load) &
                                             (ts_data_load.index < test_st_data_load)])  #[['load']]
valid = valid['load'].values
valid = valid.astype('float32')
valid = np.reshape(valid, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
valid = scaler.fit_transform(valid)

# test
test = pd.DataFrame(ts_data_load.copy().loc[(ts_data_load.index >= test_st_data_load)])  #[['load']]
test = test['load'].values
test = test.astype('float32')
test = np.reshape(test, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)



# reshape into X=t and Y=t+1
look_back = 30
X_train, Y_train = create_dataset(train, look_back)
X_valid, Y_valid = create_dataset(valid, look_back)
X_test, Y_test = create_dataset(test, look_back)


X_train.shape
Y_train.shape


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


X_train.shape


model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_valid, Y_valid), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
# Training Phase
model.summary()




# make predictions
train_predict = model.predict(X_train)
valid_predict = model.predict(X_valid)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
valid_predict = scaler.inverse_transform(valid_predict)
Y_train = scaler.inverse_transform([Y_train])
Y_valid = scaler.inverse_transform([Y_valid])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('valid Mean Absolute Error:', mean_absolute_error(Y_valid[0], valid_predict[:,0]))
print('valid Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_valid[0], valid_predict[:,0])))
print('R2:', r2_score(Y_valid[0], valid_predict[:,0]))

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='valid Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()



test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
print('R2:', r2_score(Y_test[0], test_predict[:,0]))


actual = pd.DataFrame(Y_test[0])
predict = pd.DataFrame(test_predict[:,0])
final = pd.concat([actual,predict.astype(float)],axis=1)
final.columns = ['actual','predict']

plt.figure()
plt.plot(actual, label='actual')
plt.plot(predict, label='predict')
plt.grid()
plt.legend()


def mape(ts_predictions, actuals):
    """Mean absolute percentage error"""
    return (abs(ts_predictions - actuals) / actuals).mean()


print(mape(final["predict"], final["actual"]))


(len(final[abs(final['actual']-final['predict'])<=200]))/len(final)



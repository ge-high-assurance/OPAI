#!/usr/bin/env python
# coding: utf-8
# Basic ANN model for predicting state-of-health (SOH) of Li-ion batteries
# for use in application of Overarching Properties (OP)  framework for assurance
# Uses NASA dataset referenced in Github page
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
base_dir = 'D:\\work\\FAA\\'
os.chdir(base_dir+'src\\')

TD_dir = 'D:\\work\\FAA\\data\\'

import sys
sys.path.append(base_dir+'src\\')
sys.path.append(TD_dir)

import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_data(battery):
    mat = loadmat(TD_dir + battery + '.mat')
    counter = 0
    dataset = []
    capacity_data = []

    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                          int(row['time'][0][1]),
                                          int(row['time'][0][2]),
                                          int(row['time'][0][3]),
                                          int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]
            for j in range(len(data[0][0]['Voltage_measured'][0])):
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]
                if j==0:
                    qnow=0.
                    soc=1.
                else:
                    qnow = qnow+(data[0][0]['Current_measured'][0][j] * (data[0][0]['Time'][0][j] - data[0][0]['Time'][0][j-1])) / (3600. * capacity)
                    #SOC
                    soc = 1. + qnow
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                voltage_measured, current_measured,
                                temperature_measured, current_load,
                                voltage_load, time, soc])
            capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
            counter = counter + 1
    return [pd.DataFrame(data=dataset,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity', 'voltage_measured',
                                  'current_measured', 'temperature_measured',
                                  'current_load', 'voltage_load', 'time','soc']),
            pd.DataFrame(data=capacity_data,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity'])]
def plot_data():
    plot_df = capacity.loc[(capacity['cycle'] >= 1), ['cycle', 'capacity']]
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))
    plt.plot(plot_df['cycle'], plot_df['capacity'])
    plt.ylabel('Capacity')
    adf = plt.gca().get_xaxis().get_major_formatter()
    plt.xlabel('cycle')
    plt.title('Discharge B0005')

    attrib = ['cycle', 'datetime', 'capacity']
    dis_ele = capacity[attrib]
    C = dis_ele['capacity'][0]
    for i in range(len(dis_ele)):
        dis_ele['SoH'] = (dis_ele['capacity']) / C

    plot_df = dis_ele.loc[(dis_ele['cycle'] >= 1), ['cycle', 'SoH']]
    sns.set_style("white")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df['cycle'], plot_df['SoH'])
    # Draw threshold
    plt.plot([0., len(capacity)], [0.70, 0.70])
    plt.ylabel('SOH')
    adf = plt.gca().get_xaxis().get_major_formatter()
    plt.xlabel('cycle')
    plt.title('Discharge B0005')

def soc_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=trainx.shape[1]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
    return model

def test_model(batt):
    dset, _ = load_data(batt)
    cyc = dset.cycle.values
    s = dset.soc.values

    tstx, tsty = create_training_data(dset[attrib_x],dset[attrib_y])

    pred = model.predict(tstx)

    plt.figure()
    c = np.random.randint(cyc.max())
    id = np.where((cyc == c) & (s > 0))
    plt.plot(tsty[id])
    plt.plot(pred[id])
    plt.title('Battery '+batt+' Cycle ' + str(c))
    plt.xlabel('time')
    plt.xticks([])
    plt.ylabel('SOC')
    plt.axhline(soc_th, linestyle='-.', linewidth=0.5)

    # prediction error at critical SOC value across all cycles
    rmse = []
    for i in range(cyc.max()):
        id = np.where(((cyc == i + 1) & (s <= soc_th + tol)) & (s >= soc_th - tol))
        rmse.append(np.sqrt(mean_squared_error(tsty[id], pred[id])))

    rmse = np.array(rmse)
    plt.figure()
    plt.plot(np.arange(cyc.max()), rmse)
    plt.ylabel('SOC prediction RMSE @0.3')
    plt.xlabel('Cycle')

def create_training_data(xd,yd, w=2, k=5):
    tx = xd.values
    ty = yd.values
    N = len(tx)
    for i in range(k):
        if i==0:
            tdx = tx[i*w:N-(k-i-1)*w,:]
        else:
            tmp = tx[i*w:N-(k-i-1)*w,:]
            tdx = np.concatenate((tdx,tmp),axis=1)
    tdy = ty[w*(k-1):N]
    return tdx, tdy

#main
#'B0005','B0006','B0007'
trn_batt = 'B0005'
trnflag = 0 #'0' implies standard treain-test split

attrib_x = ['voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load']

attrib_y = ['soc']

dataset, _ = load_data(trn_batt)
cycles = dataset.cycle.values
soc = dataset.soc.values

#critical low value for SOC for decision making
soc_th=0.3

#10 plots of SOC at random cycles
plt.figure()
for j in range(10):
    i = np.random.randint(150)
    print(i)
    tmp = dataset[dataset['cycle']==i]
    plt.plot(tmp.time,tmp.soc,label=str(i))
    plt.legend()

tdatax = dataset[attrib_x]
tdatay = dataset[attrib_y]

if trnflag==0:
    trainx, trainy = create_training_data(tdatax,tdatay)
    allx=trainx
    ally=trainy
else:
    #SOC level to use for partitioning train-vali
    slevel=0.3

    #training data based on level in SOC series
    idx = tdatay.soc > slevel
    tdata_x = tdatax[idx]
    tdata_y = tdatay[idx]

    #validation data as remainder of SOC series
    idx_val = tdatay.soc <= slevel
    #set aside validation data first
    vdata_x = tdatax[idx_val]
    vdata_y = tdatay[idx_val]

    #generate windowed data for model training and validation
    winsize = 2
    numwin = 5
    trainx, trainy = create_training_data(tdata_x,tdata_y,winsize, numwin)
    valx, valy = create_training_data(vdata_x,vdata_y)

    #all data
    allx, ally = create_training_data(tdatax,tdatay)
    cycles = cycles[(numwin-1)*winsize:]
    soc = soc[(numwin-1)*winsize:]

#train-test split
x_train,x_test,y_train,y_test = train_test_split(trainx,trainy,test_size=0.4)

#define and train model
model = soc_model()
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
mc = ModelCheckpoint('best_model.h5', save_best_only = True)
model.fit(x=x_train, y=y_train, batch_size=256, epochs=150, callbacks=[es,mc], validation_data=(x_test,y_test), shuffle=True)

#training quality
history = model.history
loss = history.history['loss']
val_loss = history.history['val_loss']
#epochs = range(len(loss))
epochs = range(len(loss))
plt.figure()
plt.plot(epochs,loss, label='Training loss')
plt.plot(epochs,val_loss, label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.legend()

#get best-fit model, in case of Earlystop
from keras.models import load_model
model = load_model('soc_model.h5')

#predict, visualize outcomes
tol=0.05
test_model('B0005')
test_model('B0006')
test_model('B0007')
test_model('B0018')

#save model as ONNX model
import keras2onnx
import onnx
import packaging
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model, 'soc_model.onnx')

#outcomes/predictions
for i in range(len(trn_batt)):
    batt = trn_batt[i]
    test_model(batt)

for i in range(len(tst_batt)):
    batt = tst_batt[i]
    test_model(batt)

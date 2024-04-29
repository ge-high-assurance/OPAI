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

# Function to load dataset from Matlab files into Dataframes
# from https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
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
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                voltage_measured, current_measured,
                                temperature_measured, current_load,
                                voltage_load, time])
            capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
            counter = counter + 1
    return [pd.DataFrame(data=dataset,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity', 'voltage_measured',
                                  'current_measured', 'temperature_measured',
                                  'current_load', 'voltage_load', 'time']),
            pd.DataFrame(data=capacity_data,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity'])]

#Model architecture
def soh_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=train_dataset.shape[1]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
    return model

#Model testing and outcome plots
def test_model(battnm, soh_th=0.7,verbose=1):
    #test model on different battery
    dataset_val, capacity_val = load_data(battnm)
#    soh_pred = model.predict(sc.fit_transform(dataset_val[attribs]))
    soh_pred = model.predict(dataset_val[attribs])

    C = dataset_val['capacity'][0]
    soh = []
    for i in range(len(dataset_val)):
        soh.append(dataset_val['capacity'][i] / C)
    new_soh = dataset_val.loc[(dataset_val['cycle'] >= 1), ['cycle']]
    new_soh['SoH'] = soh
    new_soh['NewSoH'] = soh_pred
    new_soh = new_soh.groupby(['cycle']).mean().reset_index()
    rms = np.sqrt(mean_squared_error(new_soh['SoH'], new_soh['NewSoH']))

    plot_df=0
    if verbose==1:
        plot_df = new_soh.loc[(new_soh['cycle'] >= 1), ['cycle', 'SoH', 'NewSoH']]
        sns.set_style("white")
        plt.figure(figsize=(16, 10))
        plt.plot(plot_df['cycle'], plot_df['SoH'], label='SoH')
        plt.plot(plot_df['cycle'], plot_df['NewSoH'], label='Predicted SoH')
        # Draw threshold
        plt.plot([0.,len(capacity_val)], [soh_th, soh_th], label='Threshold')
        plt.ylabel('SOH')
        # make x-axis ticks legible
        adf = plt.gca().get_xaxis().get_major_formatter()
        plt.xlabel('cycle')
        plt.legend()
        plt.title('Discharge '+battnm+': RMSE = '+str(rms))
    return soh_pred

#main
trn_batt=['B0005','B0006','B0007'] #batteries to train model on
tst_batt=['B0018'] #battery to test model on
attribs = ['cycle','time','voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load'] #variables used for SOH prediction

#collate training data matrix
for i in range(len(trn_batt)):
    batt = trn_batt[i]
    dataset, capacity = load_data(batt)
    C = dataset['capacity'][0]

    if i==0:
        soh = []
        for i in range(len(dataset)):
            soh.append([dataset['capacity'][i] / C])
        soh = pd.DataFrame(data=soh, columns=['SoH']).as_matrix()
        train_dataset = dataset[attribs].as_matrix()
    else:
        y = []
        for i in range(len(dataset)):
            y.append([dataset['capacity'][i] / C])
        y = pd.DataFrame(data=y, columns=['SoH'])
        train_dataset = np.concatenate((train_dataset,dataset[attribs].as_matrix()),axis=0)
        soh = np.concatenate((soh,y.as_matrix()),axis=0)

#train-test split
x_train, x_test, y_train, y_test = train_test_split(train_dataset, soh, test_size=0.1)

#define model
model = soh_model()

#TRAIN
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
mc = ModelCheckpoint('best_model.h5', save_best_only = True)
model.fit(x=x_train, y=y_train, batch_size=512, epochs=150, callbacks=[es,mc], validation_data=(x_test,y_test), shuffle=True)

#evaluate training quality
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

model = load_model('best_model.h5')

#evaluate outcomes/predictions on test battery dataset
for i in range(len(tst_batt)):
    batt = tst_batt[i]
    test_model(batt)

#save model as ONNX model
import keras2onnx
import onnx
import packaging
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model, 'soh_model.onnx')


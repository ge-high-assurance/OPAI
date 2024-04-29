from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

battnm = 'B0018'

def load_data(battery):
    mat = loadmat(battery + '.mat')
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

def return_dset(battnm):
    dataset_val, capacity_val = load_data(battnm)

    attribs = ['cycle','time','voltage_measured', 'current_measured',
            'temperature_measured', 'current_load', 'voltage_load'] #variables used for SOH prediction

    C = dataset_val['capacity'][0]
    soh = []
    for i in range(len(dataset_val)):
        soh.append(dataset_val['capacity'][i] / C)
    soh = pd.DataFrame(data=soh, columns=['SoH'])
    print(soh)
    train_dataset = dataset_val[attribs]
    return train_dataset

if __name__ == "__main__":
    trn_batt = ['B0005','B0006','B0007']
    for i in range(len(trn_batt)):
        batt = trn_batt[i]
        dataset, capacity = load_data(batt)
        C = dataset['capacity'][0]

        attribs = ['cycle','time','voltage_measured', 'current_measured',
            'temperature_measured', 'current_load', 'voltage_load'] #variables used for SOH prediction

        if i==0:
            soh = []
            for i in range(len(dataset)):
                soh.append([dataset['capacity'][i] / C])
            soh = pd.DataFrame(data=soh, columns=['SoH']).to_numpy()
            train_dataset = dataset[attribs].to_numpy()
        else:
            y = []
            for i in range(len(dataset)):
                y.append([dataset['capacity'][i] / C])
            y = pd.DataFrame(data=y, columns=['SoH'])
            train_dataset = np.concatenate((train_dataset,dataset[attribs].to_numpy()),axis=0)
            soh = np.concatenate((soh,y.to_numpy()),axis=0)
        
    df = pd.DataFrame(train_dataset, columns=attribs)
    df.to_csv("soh_train_data_x.csv")
    df = pd.DataFrame(soh, columns=["SoH"])
    df.to_csv("soh_train_data_y.csv")
    print(df)

    


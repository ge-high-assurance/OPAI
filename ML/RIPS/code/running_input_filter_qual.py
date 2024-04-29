import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

import pickle

def run_input_filter(x):
    with open('input_filter.pkl', 'rb') as f:
        gm = pickle.load(f)

    #print((gm.score_samples(x[:,0:5]) > -42.7408).sum()/len(x[:,0:5]))
    
    return (gm.score_samples(x[:,0:5]) > -42.7408)

if __name__ == "__main__":
    qual_df = pd.read_csv("soh_qual_data_x.csv")
    train_df = pd.read_csv("soh_train_data_x.csv")

    qual_data = qual_df.to_numpy()
    train_data = train_df.to_numpy()

    #train_input_filter()
    print("Fraction of points in test data passed:")
    print(run_input_filter(qual_data).sum()/len(qual_data))
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

import pickle

def train_input_filter():
    qual_df = pd.read_csv("soh_qual_data_x.csv")
    train_df = pd.read_csv("soh_train_data_x.csv")

    qual_data = qual_df.to_numpy()
    train_data = train_df.to_numpy()

    best_val = -1
    best_n_comp = -1

    for i in range(1,7):
        gm = GaussianMixture(n_components=i, random_state=0, max_iter = 10000).fit(train_data[:,0:5])
        aic_cur = gm.aic(train_data[:,0:5])
        print(aic_cur)
        if best_val == -1:
            best_val = aic_cur
            best_n_comp = i
        elif aic_cur < best_val:
            best_val = aic_cur
            best_n_comp = i

    gm = GaussianMixture(n_components=best_n_comp, random_state=0, max_iter = 1000).fit(train_data[:,0:5])

    #gm = GaussianMixture(n_components=best_n_comp, random_state=0, max_iter = 1000).fit(train_data[:,0:5])
    #print(min(gm.score_samples(train_data[:,0:5])))
    #print(gm.score_samples(qual_data[:,0:5]))

    #print(((gm.score_samples(qual_data[:,0:5]) > min(gm.score_samples(train_data[:,0:5]))).sum()/len(qual_data[:,0:5])))

    with open('input_filter.pkl', 'wb') as f:
        pickle.dump(gm,f)
    

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
    print("Number of points in test data passed:")
    print(run_input_filter(qual_data).sum()/len(qual_data))

    


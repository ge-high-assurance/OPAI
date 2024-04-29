from input_filter import run_input_filter
import numpy as np
import pandas as pd

ranges = [[0,200],[0,3.6e+7],[0,40],[0,3],[-15,55],[0,3],[0,40]]
np.random.seed(0)
off_nominal_dataset = np.random.rand(500000,7)

for i in range(len(ranges)):
    off_nominal_dataset[:,i] = off_nominal_dataset[:,i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]

off_nominal_labels = np.random.rand(500000,1)

np.save("off_nominal.npy", off_nominal_dataset)

df = pd.DataFrame(off_nominal_dataset)


df.to_csv("off_nominal_x.csv", header=False, index=False)



df = pd.DataFrame(off_nominal_labels)
df.to_csv("off_nominal_y.csv", header=False, index=False)



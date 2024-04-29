from input_filter import run_input_filter
import numpy as np

ranges = [[0,200],[0,3.6e+7],[0,40],[0,3],[-15,55],[0,3],[0,40]]
np.random.seed(0)
off_nominal_dataset = np.random.rand(500000,7)

for i in range(len(ranges)):
    off_nominal_dataset[:,i] = off_nominal_dataset[:,i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]

np.save("off_nominal.npy", off_nominal_dataset)

filtered_dataset = off_nominal_dataset[run_input_filter(off_nominal_dataset)]
print(len(filtered_dataset)/len(off_nominal_dataset))
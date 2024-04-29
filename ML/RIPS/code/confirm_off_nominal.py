from input_filter import run_input_filter
import numpy as np

off_nominal_dataset = np.load("off_nominal.npy")

filtered_dataset = off_nominal_dataset[run_input_filter(off_nominal_dataset)]
print(len(filtered_dataset)/len(off_nominal_dataset))
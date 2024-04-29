import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance

x_qual_df = pd.read_csv("soh_qual_data_x.csv")
x_qual_np = x_qual_df.to_numpy()
y_qual_df = pd.read_csv("soh_qual_data_y.csv")
y_qual_np  = y_qual_df.to_numpy()

x_train_df = pd.read_csv("soh_train_data_x.csv")
x_train_np = x_train_df.to_numpy()
y_train_df = pd.read_csv("soh_train_data_y.csv")
y_train_np = y_qual_df.to_numpy()

print("mannwhitney results:")
for i in range(len(x_qual_np[0])):
    print("feature: ", i)
    _, p = mannwhitneyu(x_qual_np[i], x_train_np[i])
    print("similarity likelihood: ", p)
    print()

print("label")
_, p = mannwhitneyu(y_qual_np[i], y_train_np[i])
print("similarity likelihood: ", p)

print("Kolmogorov-Smirnov results:")
for i in range(len(x_qual_np[0])):
    print("feature: ", i)
    _, p = ks_2samp(x_qual_np[i], x_train_np[i])
    print("similarity likelihood: ", p)
    print()

print("label")
_, p = ks_2samp(y_qual_np[i], y_train_np[i])
print("similarity likelihood: ", p)

print("Wasserstein distance results:")
for i in range(len(x_qual_np[0])):
    print("feature: ", i)
    w = wasserstein_distance(x_qual_np[i], x_train_np[i])
    print("weight distance: ", w)
    print()

print("label")
w = wasserstein_distance(y_qual_np[i], y_train_np[i])
print("weight distance: ", p)

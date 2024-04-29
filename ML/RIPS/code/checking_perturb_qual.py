import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np
from input_filter import run_input_filter


#running model
networkName = "../soh_model.onnx"

x_df = pd.read_csv("soh_qual_data_x.csv")
#print(x_df)
x_np = x_df.to_numpy()
x_np = x_np.astype(np.float32)

y_df = pd.read_csv("soh_qual_data_y.csv")
y_label = y_df.to_numpy()


session = onnxruntime.InferenceSession(networkName, None)
inputName = session.get_inputs()[0].name
outputName = session.get_outputs()[0].name
output = session.run([outputName], {inputName: x_np})

#doing an analysis on 2% perturbation
output = output[0]
#print(y_label)
print((np.logical_and((output >= y_label*0.98), (output <= y_label*1.02))).sum()/len(y_label))



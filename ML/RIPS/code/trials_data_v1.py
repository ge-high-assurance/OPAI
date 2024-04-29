import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np

x_df = pd.read_csv("soh_qual_data_x.csv")
print(x_df)
x_np = x_df.to_numpy()
x_cycle = x_np[:,0]
plt.hist(x_cycle, bins=50)
plt.title("Cycle")
plt.xlabel("Cycle value")
plt.ylabel("Frequency")
plt.savefig("cycles_hist.png")
plt.clf()

x_time = x_np[:,1]
plt.hist(x_time, bins=50)
plt.title("Time")
plt.xlabel("Time value")
plt.ylabel("Frequency")
plt.savefig("time_hist.png")
plt.clf()

x_volt = x_np[:,2]
plt.hist(x_volt, bins=50)
plt.title("Voltage measured")
plt.xlabel("Voltage measured value")
plt.ylabel("Frequency")
plt.savefig("volt_hist.png")
plt.clf()

x_current = x_np[:,3]
plt.hist(x_current, bins=50)
plt.title("Current measured")
plt.xlabel("Current measured value")
plt.ylabel("Frequency")
plt.savefig("current_hist.png")
plt.clf()

x_temp = x_np[:,4] 
plt.hist(x_temp, bins=50)
plt.title("Temperature measured")
plt.xlabel("Temperature measured value")
plt.ylabel("Frequency")
plt.savefig("temperature_hist.png")
plt.clf()

x_current_load = x_np[:,5] 
plt.hist(x_temp, bins=50)
plt.title("Current load measured")
plt.xlabel("Current load measured value")
plt.ylabel("Frequency")
plt.savefig("current_load_hist.png")
plt.clf()

x_voltage_load = x_np[:,6] 
plt.hist(x_temp, bins=50)
plt.title("Voltage load measured")
plt.xlabel("Voltage load measured value")
plt.ylabel("Frequency")
plt.savefig("voltage_load_hist.png")
plt.clf()

y_df = pd.read_csv("soh_qual_data_y.csv")
y_label = y_df.to_numpy()
plt.hist(y_label[:,0], bins=50)
plt.title("SoH measured") 
plt.xlabel("SoH measured value")
plt.ylabel("Frequency")
plt.savefig("SoHloadhist.png")
plt.clf()

#running model
networkName = "../soh_model.onnx"
x_np = x_np.astype(np.float32)
print(x_np)

session = onnxruntime.InferenceSession(networkName, None)
inputName = session.get_inputs()[0].name
outputName = session.get_outputs()[0].name
output = session.run([outputName], {inputName: x_np})
print(output)
print(y_label)
print((output == y_label).sum())
print(abs(output-y_label).sum()/len(output))
print(np.power(output-y_label,2).sum()/len(output))



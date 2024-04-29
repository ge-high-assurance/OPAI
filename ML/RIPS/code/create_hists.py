import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np
from input_filter import run_input_filter

x_df = pd.read_csv("soh_qual_data_x.csv")
print(x_df)
x_np = x_df.to_numpy()
x_cycle = x_np[:,0]
plt.hist(x_cycle, bins=50)
plt.title("Cycle")
plt.xlabel("Cycle value (Units)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,200)
print("cycle in range:", str(np.bitwise_and((x_cycle >=0),(x_cycle <= 200)).sum()/len(x_cycle)))
plt.savefig("cycles_hist.png")
plt.clf()

x_time = x_np[:,1]
plt.hist(x_time, bins=50)
plt.title("Time")
plt.xlabel("Time value (Seconds)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,3.6e+7)
print("time in range:", str(np.bitwise_and((x_time >=0),(x_time <= 3.6e+7)).sum()/len(x_time)))
plt.savefig("time_hist.png")
plt.clf()

x_volt = x_np[:,2]
plt.hist(x_volt, bins=50)
plt.title("Voltage measured")
plt.xlabel("Voltage measured value (Vdc)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,40)
print("time in range:", str(np.bitwise_and((x_volt >=0),(x_volt <= 40)).sum()/len(x_volt)))
plt.savefig("volt_hist.png")
plt.clf()

x_current = x_np[:,3]
plt.hist(x_current, bins=50)
plt.title("Current measured")
plt.xlabel("Current measured value (A)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,3)
print("current in range:", str(np.bitwise_and((x_current >=0),(x_current <= 3)).sum()/len(x_current)))
plt.savefig("current_hist.png")
plt.clf()

x_temp = x_np[:,4] 
plt.hist(x_temp, bins=50)
plt.title("Temperature measured")
plt.xlabel("Temperature measured value (Celsius)")
plt.ylabel("Frequency (Units)")
plt.xlim(-15,55)
print("temperature in range:", str(np.bitwise_and((x_temp >=-15),(x_temp <= 55)).sum()/len(x_temp)))
plt.savefig("temperature_hist.png")
plt.clf()

x_current_load = x_np[:,5] 
plt.hist(x_temp, bins=50)
plt.title("Current load measured")
plt.xlabel("Current load measured value (A)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,3)
print("current in range:", str(np.bitwise_and((x_current_load >=0),(x_current_load <= 3)).sum()/len(x_current_load)))
plt.savefig("current_load_hist.png")
plt.clf()

x_voltage_load = x_np[:,6] 
plt.hist(x_temp, bins=50)
plt.title("Voltage load measured")
plt.xlabel("Voltage load measured value (A)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,40)
print("voltage in range:", str(np.bitwise_and((x_voltage_load >=0),(x_voltage_load <= 40)).sum()/len(x_voltage_load)))
plt.savefig("voltage_load_hist.png")
plt.clf()

y_df = pd.read_csv("soh_qual_data_y.csv")
y_label = y_df.to_numpy()
plt.hist(y_label[:,0], bins=50)
plt.title("SoH measured") 
plt.xlabel("SoH measured value (Ratio)")
plt.ylabel("Frequency (Units)")
plt.xlim(0,1)
print("SoH in range:", str(np.bitwise_and((y_label >=0),(y_label <= 1)).sum()/len(y_label)))
plt.savefig("SoHloadhist.png")
plt.clf()
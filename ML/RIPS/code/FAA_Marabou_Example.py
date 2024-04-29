import math
import numpy as np
import time

import sys
sys.path.append("/Users/michaeldurling/git/Marabou")

from maraboupy import Marabou
from maraboupy import MarabouCore

"""
# Read in the neural network file
"""

options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=120, \
    numWorkers=1, solveWithMILP=False)
filename = "./soh_model.onnx"
network = Marabou.read_onnx(filename)

print('\n')

print('Network Name =',filename,'\n')

"""
# Get the input and output variable numbers; [0] since first dimension is batch size
"""
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

print("Input Variables = ",inputVars)
print("Output Variables = ",outputVars,'\n')

"""
attribs = ['cycle','time','voltage_measured', 'current_measured',
           'temperature_measured', 'current_load', 'voltage_load'] #variables used for SOH prediction

Voltage_measured: Battery terminal voltage (Volts); range = [1.73703003, 4.23332544]

Current_measured: Battery output current (Amps); range = [-2.02909849, 8.80861221e-03]

Temperature_measured: Battery temperature (degree C); range = [22.96992298, 4.23325224e+01]

Current_load: Current measured at load (Amps); range = [-2, 2]

Voltage_load: Voltage measured at load (Volts); range = [0, 4.249]

Time: Time vector for the cycle (secs); range = [0, 3.69023400e+03]

Cycle: the charge-discharge cycle number; range = [1, 168]
        (each cycle starts with a charging, followed by discharging cycle, as were described above)

"""


"""
# Set input bounds
"""
network.setLowerBound(inputVars[0], 1.0)
network.setUpperBound(inputVars[0], 200.0)
network.setLowerBound(inputVars[1], 0.0)
network.setUpperBound(inputVars[1], 5e+03)
network.setLowerBound(inputVars[2], 2.75)
network.setUpperBound(inputVars[2], 4.2)
network.setLowerBound(inputVars[3], -3.0)
network.setUpperBound(inputVars[3], 0)
network.setLowerBound(inputVars[4], 20)
network.setUpperBound(inputVars[4], 45)
network.setLowerBound(inputVars[5], -3.0)
network.setUpperBound(inputVars[5], 3.0)
network.setLowerBound(inputVars[6], 0.0)
network.setUpperBound(inputVars[6], 5.0)

"""
# Set output bounds
"""
network.setLowerBound(outputVars[0], 0.0)
network.setUpperBound(outputVars[0], 0.5)

"""
# Call to Marabou solver
"""
exitCode, vals, stats = network.solve(options = options)
#assert( exitCode == "unsat")
#assert len(vals) == 0

print('\n')

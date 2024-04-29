import numpy as np
import time
import onnxruntime
import looking_data
import pandas as pd

from maraboupy import Marabou
from maraboupy import MarabouCore

def run_simulation_soh(state):
    networkName = "../soh_model.onnx"

    state = np.array(state, dtype=np.float32)

    session = onnxruntime.InferenceSession(networkName, None)
    inputName = session.get_inputs()[0].name
    outputName = session.get_outputs()[0].name
    output = session.run([outputName], {inputName: state[None,:]})[0][0]
    return output

def attempt_marabou_soh_perturb(point, input_eps, output_eps):
    start = time.perf_counter()
    options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=120, \
    numWorkers=10, solveWithMILP=False)

    networkName = "../soh_model.onnx"

    network = Marabou.read_onnx(networkName)

    input = network.inputVars[0][0]
    output = network.outputVars[0][0]

    '''
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
    '''

    network.setLowerBound(input[0], point[0] - input_eps)
    network.setUpperBound(input[0], point[0] + input_eps)
    network.setLowerBound(input[1], point[1] - input_eps)
    network.setUpperBound(input[1], point[1] + input_eps)
    network.setLowerBound(input[2], point[2] - input_eps)
    network.setUpperBound(input[2], point[2] + input_eps)
    network.setLowerBound(input[3], point[3] - input_eps)
    network.setUpperBound(input[3], point[3] + input_eps)
    network.setLowerBound(input[4], point[4] - input_eps)
    network.setUpperBound(input[4], point[4] + input_eps)
    network.setLowerBound(input[5], point[5] - input_eps)
    network.setUpperBound(input[5], point[5] + input_eps)
    network.setLowerBound(input[6], point[6] - input_eps)
    network.setUpperBound(input[6], point[6] + input_eps)

    '''
    e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
    e1.addAddend(1.0, output[0])
    e1.setScalar(1.0768)

    e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
    e2.addAddend(1.0, output[0])
    e2.setScalar(1.077)

    '''

    #enforce that the output is strictly larger than 0.999
    e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
    e1.addAddend(1.0, output[0])
    bound = 1 - output_eps
    e1.setScalar(bound)

    #enforce output is strictly less than 1.001
    e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
    e2.addAddend(1.0, output[0])
    bound = 1.0 + output_eps
    e2.setScalar(bound)

    #disjunct over all constraints to form the negation of the output values
    #checking that output in between 0.7 and 1.0
    network.addDisjunctionConstraint([[e1],[e2]])


    #finally solve
    exitCode, vals, stats = network.solve(options = options, verbose = True)
    end = time.perf_counter()
    if exitCode == "sat":
        #print the cycle if there is a counterexample
        print("Counterexample found for input", vals[input[0]])
    else:
        print("Proof completed")
    print("Time taken:", end-start)
    return exitCode

def attempt_marabou_soh_perturb_percent(point, soh_val, input_perc, output_perc):
    start = time.perf_counter()
    options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=120, \
    numWorkers=10, solveWithMILP=False)

    networkName = "../soh_model.onnx"

    network = Marabou.read_onnx(networkName)

    input = network.inputVars[0][0]
    output = network.outputVars[0][0]

    '''
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
    '''

    network.setLowerBound(input[0], point[0] - point[0]*input_perc)
    network.setUpperBound(input[0], point[0] + point[0]*input_perc)
    network.setLowerBound(input[1], point[1] - point[1]*input_perc)
    network.setUpperBound(input[1], point[1] + point[1]*input_perc)
    network.setLowerBound(input[2], point[2] - point[2]*input_perc)
    network.setUpperBound(input[2], point[2] + point[2]*input_perc)
    network.setLowerBound(input[3], point[3] - point[3]*input_perc)
    network.setUpperBound(input[3], point[3] + point[3]*input_perc)
    network.setLowerBound(input[4], point[4] - point[4]*input_perc)
    network.setUpperBound(input[4], point[4] + point[4]*input_perc)
    network.setLowerBound(input[5], point[5] - point[5]*input_perc)
    network.setUpperBound(input[5], point[5] + point[5]*input_perc)
    network.setLowerBound(input[6], point[6] - point[6]*input_perc)
    network.setUpperBound(input[6], point[6] + point[6]*input_perc)

    '''
    e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
    e1.addAddend(1.0, output[0])
    e1.setScalar(1.0768)

    e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
    e2.addAddend(1.0, output[0])
    e2.setScalar(1.077)

    '''

    #enforce that the output is strictly larger than 0.999
    e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
    e1.addAddend(1.0, output[0])
    bound = soh_val - soh_val*output_perc
    e1.setScalar(bound)

    #enforce output is strictly less than 1.001
    e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
    e2.addAddend(1.0, output[0])
    bound = soh_val + soh_val*output_perc
    e2.setScalar(bound)

    #disjunct over all constraints to form the negation of the output values
    #checking that output in between 0.7 and 1.0
    network.addDisjunctionConstraint([[e1],[e2]])


    #finally solve
    exitCode, vals, stats = network.solve(options = options, verbose = True)
    end = time.perf_counter()
    if exitCode == "sat":
        #print the cycle if there is a counterexample
        print("Counterexample found for input", vals[input[0]])
    else:
        print("Proof completed")
    print("Time taken:", end-start)
    return exitCode


def find_range(dataset):
    start = time.perf_counter()
    output_range = 0.01
    input_ranges = [x*0.0001 for x in range(10)]
    outputs = np.zeros(len(dataset))
    for i in range(len(dataset)):
        for input_range in input_ranges:
            res = attempt_marabou_soh_perturb(dataset[i],input_range,output_range)
            if res == "sat":
                break
            else:
                outputs[i] = input_range
    end = time.perf_counter()
    print("Time taken for ranges:", end-start)
    return outputs



if __name__ == "__main__":
    #grab data

    x_df = pd.read_csv("soh_train_data_x.csv")
    print(len(x_df))
    num_sat = 0
    num_ok = 0
    output_df = pd.read_csv("soh_train_data_y.csv")
    x_np = x_df.to_numpy()
    output_np = output_df.to_numpy()
    for i in range(len(x_np)):
        exCode = attempt_marabou_soh_perturb_percent(x_np[i,:], output_np[i,:][0], 0.04, 0.02)
        if exCode == "sat":
            num_sat += 1
        elif exCode == "unsat":
            num_ok += 1
    
    print((num_ok)/(num_ok+num_sat))
    
    quit()

    run_simulation_soh
    x_df = pd.read_csv("soh_qual_data_x.csv")
    x_np = x_df.to_numpy()

    print(len(x_df))
    output_df = pd.read_csv("soh_qual_data_y.csv")
    output_np = output_df.to_numpy()
    #first point
    for i in range(len(x_np)):
        exCode = attempt_marabou_soh_perturb_percent(x_np[i,:], output_np[i,:][0], 0.04, 0.02)
        if exCode == "sat":
            break
            print(i)
    
    
    x_df = pd.read_csv("soh_train_data_x.csv")
    print(len(x_df))
    output_df = pd.read_csv("soh_train_data_y.csv")
    x_np = x_df.to_numpy()
    output_np = output_df.to_numpy()
    for i in range(len(x_np)):
        exCode = attempt_marabou_soh_perturb_percent(x_np[i,:], output_np[i,:][0], 0.04, 0.02)
        if exCode == "sat":
            break
            print(i)
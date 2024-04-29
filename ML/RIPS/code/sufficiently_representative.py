import os
import sys
import errno
import shutil
import random
import numpy as np
import csv



#--------------- Functions
#-- Function that takes a range and a value and checks if the value is within that range
def valueInRange(floor, ceiling, value):
    if((value >= floor) and (value <= ceiling)):
        return True
    else:
        return False

#-- Function that takes a range and a set of values and checks if there is at least one value in the range
def checkRangeCoverage(floor, ceiling, values):
    coverageFlag = False
    for x in values:
        if(valueInRange(floor, ceiling, x)):
            coverageFlag = True
            print ("Between",  floor, " and ", ceiling, " ", x, " gives ", coverageFlag)
    return coverageFlag

#-- Function that takes a single range of a variable as assigned to a bin, a radius, and a set of values for a variable and checks if the variable is sufficently spread
def checkVariableBinCoverage(floor,ceiling,radius,values):
    currentFloor = floor
    currentCeiling = floor + radius
    coverageFlag = True
    # keep varying the range in radius sized chunks and checking
    # Note: the last currentCeiling may not cover the actual ceiling, so have to check separately
    while((currentFloor < ceiling) and (currentCeiling < ceiling)):
        # if the current range chunk does not contain a value, turn flag to false and break
        if(checkRangeCoverage(currentFloor, currentCeiling, values) == False):
            coverageFlag = False
            print ("Nothing found between",  currentFloor, " and ", currentCeiling)
            break
        # otherwise, update to next range chunk
        else:
            currentFloor = currentCeiling
            currentCeiling = currentCeiling + radius
    # if coverageFlag has not been assigned false, it means that the entire while loop was completed
    # so we check the last range betwee currentCeiling and ceiling here
    if coverageFlag:
        if(checkRangeCoverage(currentCeiling, ceiling, values)):
            coverageFlag = False
            print ("Nothing found between",  currentCeiling, " and ", ceiling)
    
    return coverageFlag 

#-- Function that takes packaged data for a variable and checks variable coverage
def checkVariableCoverageFromData(rangePair, nameRadiusValueTuple):
    return(checkVariableBinCoverage(rangePair[0], rangePair[1], nameRadiusValueTuple[1], nameRadiusValueTuple[2]))


#-- Function that takes a bin and variable details and checks multivariate bin coverage
#-- bin: [(floor,ceiling),(),(),(),...]
#-- varDetails: [(varName,radius,values),(),(),(),...] where values: [v,....]
#-- Note: bin and varDetails must have same no. of elements
def checkMultivariateBinCoverage(bin, varDetails):
    binCoverage = True
    # for each elemnt in the bin, which corresponds to a variable range, the variable coverage must be true
    for i in range (0, len(bin)): 
        print("------ VARIABLE: ", varDetails[i][0])
        # if variable coverage is not met for this variable, set flag to false and break
        if(checkVariableCoverageFromData(bin[i], varDetails[i]) == False):
            binCoverage = False
            break
    return binCoverage



#-- testing
# print(checkVariableBinCoverage(0, 100, 10 , [2, 4, 15, 26, 34, 37, 47, 63, 71, 89, 93]))
# bin = [(0,30),(.1, .5),(-60, 0)]
# varDetails = [("cycle",10,[5,15,25]),("time",.1,[.13,.23,.33,.43]),("voltage",20,[-55, -35, -15])]
# print(checkMultivariateBinCoverage(bin, varDetails))

#-- set bin from requirements
#-- [vol_m, cur_m, tmp_m, cur_l, vol_l]
bin = [(0, 32),(0, 3),(-15, 55),(0, 3),(0, 32)]
#-- get variable data from csv files
vol_m=[]
cur_m=[]
tmp_m=[]
cur_l=[]
vol_l=[]
with open("../data/soh_qual_data_x.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)
    for row in readCSV:
        vol_m.append(float(row[2]))
        cur_m.append(float(row[3]))
        tmp_m.append(float(row[4]))
        cur_l.append(float(row[5]))
        vol_l.append(float(row[6]))
#-- set varDetails
varDetails = [("vol_m",0.1,vol_m),("cur_m",0.1,cur_m),("tmp_m",1,tmp_m),("cur_l",0.1,cur_l),("vol_l",0.1,vol_l)]
#-- run
print(checkMultivariateBinCoverage(bin, varDetails))

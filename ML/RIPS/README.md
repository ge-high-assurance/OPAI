# Provenance of artifacts in this directory


data:
1.	soh_qual_data_x.csv
   - Description: Contains the features for the test dataset of the AC-1.h5
   - Source: B0018.mat
3.	soh_qual_data_y.csv
   - Description: Contains the labels for the test dataset of the AC-1.h5
   - Source: B0018.mat
5.	soh_train_data_x.csv
   - Description: Contains the features for the training dataset of the AC-1.h5
   - Source: B0005.mat, B0006.mat, B0007.mat
6.	soh_train_data_y.csv
   - Description: Contains the labels for the training dataset of the AC-1.h5
   - Source: B0005.mat, B0006.mat, B0007.mat
7.	B0005.mat
   - Description: The data for one battery. Includes and features and labels trained on.
   - Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
8.	B0006.mat
   - Description: The data for one battery. Includes and features and labels trained on.
   - Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
9.	B0007.mat
   - Description: The data for one battery. Includes and features and labels trained on.
   - Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
10.	B00018.mat
   - Description: The data for one battery.  Includes and features and labels trained on.
   - Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
11.	off_nominal_x.csv
   - Description: Generated off nominal points, features only.
   - Source: N/A, generated randomly from code.
12.	off_nominal_y.csv
   - Description: Generated off nominal points, labels only.
   - Source: N/A, generated randomly from code.
13.	off_nominal.npy
   - Description: Generated off nominal points, features only in numpy format.
   - Source: N/A, generated randomly from code.


model:
1.	soh_model.h5
   - Description: The SOH model
   - Dataset used to create: B0005.mat, B0006.mat, B0007.mat
   - Code used to create: battery_soh.py
3.	soh_model.onnx
   - Description: The SOH model
   - Dataset used to create: N/A (conversion from soh_model.h5)
   - Code used to create: h5_to_onnx_v1.py
4.	trn_gmm.mat
   - Description: The GMM model trained using MATLAB
   - Dataset used to create: soh_train_data_x.csv
   - Code used to create: gmm_script.gmm
5.	Input_filer.pkl
   - Description: The GMM model trained using Python
   - Dataset used to create: soh_train_data_x.csv
   - Code used to create: input_filter.py


code:
1.	test_off_nominal.py
   - Description: Checks if all off nominal inputs pass the input filter, and saves and generates this data.
2.	training_perturb.py
   - Description: Passes the training data to Marabou to check robustness (by seeing if having 4% of perturbation in input causes more than 2% of perturbation in output).
3.	confirm_off_nominal.py
   - Description: Tests if all off nominal inputs generated do not pass the input filter.
4.	generate_off_nominal.py
   - Description: Generates off nominal inputs to test the input filter with.
5.	running_input_filter_qual.py
   - Description: Runs the input filter on all qualification data to see if it passes.
6.	checking_perturb_qual.py
   - Description: Checks the number of test data which does not produce output more than 2% away from its label.
7.	checking_filter_perturb_qual.py
   - Description: Checks the number of passing test data which does not produce output more than 2% away from its label.
8.	gmm_train.m,gmm_script.m,gmm_input_filter.m
   - Description: Matlab code for training the input filter (by Naresh).
9.	input_filter.py
   - Description: Trains the input filter in python.
10.	soh_playground.py
   - Description: Passes the training data to Marabou to check robustness (by seeing if having 4% of perturbation in input causes more than 2% of perturbation in output).
11.	trials_data.py
   - Description: Generates histograms and statistics on each feature checking distributions and if they fall in expected ranges, for qual data.
12.	looking_data.py
   - Description: Converts mat data files to csv data files, and can help with looking into the datasets with print statements.
13.	data_similarity_analysis.py
   - Description: Statistical tests to assess similarity between training and testing data (I.e. Mann-Whitney similarity test)
14.	battery_soh.py
   - Description: Trains the soh model (in h5 format).
15.	create_hists.py
   - Description: Generates histograms and statistics on each feature checking distributions and if they fall in expected ranges, for qual data.

<hr>
Copyright (c) 2021-2024 General Electric Company

All Rights Reserved

This material is based upon work supported by the Federal Aviation Administration (FAA) under Contract No. 692M15-22-T-00012.

This website represents research work funded by the Federal Aviation Administration (FAA) and it is disseminated under the sponsorship of the U.S. Department of Transportation in the interest of information exchange. The U.S. Government assumes no liability for the contents or use thereof. The U.S. Government does not endorse products or manufacturers. Trade or manufacturers’ names appear herein solely because they are considered essential to the objective of this presentation/paper. The findings and conclusions are those of the author(s) and do not necessarily represent the views of the funding agency. This document does not constitute FAA policy.



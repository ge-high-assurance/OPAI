# Provenance of artifacts in this directory


data:
1.	soh_qual_data_x.csv
	a.	Description: Contains the features for the test dataset of the AC-1.h5 
	b.	Source: B0018.mat
2.	soh_qual_data_y.csv
	a.	Description: Contains the labels for the test dataset of the AC-1.h5
	b.	Source: B0018.mat
3.	soh_train_data_x.csv
	a.	Description: Contains the features for the training dataset of the AC-1.h5 
	b.	Source: B0005.mat, B0006.mat, B0007.mat
4.	soh_train_data_y.csv
	a.	Description: Contains the labels for the training dataset of the AC-1.h5
	b.	Source: B0005.mat, B0006.mat, B0007.mat
5.	B0005.mat
	a.	Description: The data for one battery. Includes and features and labels trained on.
	b.	Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
6.	B0006.mat
	a.	Description: The data for one battery. Includes and features and labels trained on.
	b.	Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
7.	B0007.mat
	a.	Description: The data for one battery. Includes and features and labels trained on.
	b.	Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
8.	B00018.mat
	a.	Description: The data for one battery.  Includes and features and labels trained on.
	b.	Source: NASA Repo, https://github.com/psanabriaUC/BatteryDatasetImplementation/tree/master/battery_data
9.	off_nominal_x.csv
	a.	Description: Generated off nominal points, features only.
	b.	Source: N/A, generated randomly from code.
10.	off_nominal_y.csv
	a.	Description: Generated off nominal points, labels only.
	b.	Source: N/A, generated randomly from code.
11.	off_nominal.npy
	a.	Description: Generated off nominal points, features only in numpy format.
	b.	Source: N/A, generated randomly from code.


model:
1.	soh_model.h5
	a.	Description: The SOH model
	b.	Dataset used to create: B0005.mat, B0006.mat, B0007.mat
	c.	Code used to create: battery_soh.py
2.	soh_model.onnx
	a.	Description: The SOH model
	b.	Dataset used to create: N/A (conversion from soh_model.h5)
	c.	Code used to create: h5_to_onnx_v1.py
3.	trn_gmm.mat
	a.	Description: The GMM model trained using MATLAB
	b.	Dataset used to create: soh_train_data_x.csv
	c.	Code used to create: gmm_script.gmm
4.	Input_filer.pkl
	a.	Description: The GMM model trained using Python
	b.	Dataset used to create: soh_train_data_x.csv
	c.	Code used to create: input_filter.py


code:
1.	test_off_nominal.py
	a.	Description: Checks if all off nominal inputs pass the input filter, and saves and generates this data.
2.	training_perturb.py
	a.	Description: Passes the training data to Marabou to check robustness (by seeing if having 4% of perturbation in input causes more than 2% of perturbation in output).
3.	confirm_off_nominal.py
	a.	Description: Tests if all off nominal inputs generated do not pass the input filter.
4.	generate_off_nominal.py
	a.	Description: Generates off nominal inputs to test the input filter with.
5.	running_input_filter_qual.py
	a.	Description: Runs the input filter on all qualification data to see if it passes.
6.	checking_perturb_qual.py
	a.	Description: Checks the number of test data which does not produce output more than 2% away from its label.
7.	checking_filter_perturb_qual.py
	a.	Description: Checks the number of passing test data which does not produce output more than 2% away from its label.
8.	gmm_train.m,gmm_script.m,gmm_input_filter.m
	a.	Description: Matlab code for training the input filter (by Naresh).
9.	input_filter.py
	a.	Description: Trains the input filter in python.
10.	soh_playground.py
	a.	Description: Passes the training data to Marabou to check robustness (by seeing if having 4% of perturbation in input causes more than 2% of perturbation in output).
11.	trials_data.py
	a.	Description: Generates histograms and statistics on each feature checking distributions and if they fall in expected ranges, for qual data.
12.	looking_data.py
	a.	Description: Converts mat data files to csv data files, and can help with looking into the datasets with print statements.
13.	data_similarity_analysis.py
	a.	Description: Statistical tests to assess similarity between training and testing data (I.e. Mann-Whitney similarity test)
14.	battery_soh.py
	a.	Description: Trains the soh model (in h5 format).
15.	create_hists.py
	a.	Description: Generates histograms and statistics on each feature checking distributions and if they fall in expected ranges, for qual data.





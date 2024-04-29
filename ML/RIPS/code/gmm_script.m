%read SOH data
trndata = csvread('soh_train_data_x.csv',1,0);
qualdata = csvread('soh_qual_data_x.csv',1,0);

%train GMM to estimate modes 'k' and likelihood threshold nl_th
[gmm_model,q,n] = gmm_train(trndata(:,1:5), qualdata(:,1:5),1,[1,6]);
%settled for k=5, nl_th = 42.

%save GMM model for application as input filter
save('trn_gmm.mat', 'gmdl')

%apply input filter to qual-data
qfilt = gmm_input_filter(qualdata(:,1:5),gmm.gmdl);






%%%
%%%
% Simpler examples

%Example 1    
n1 = normrnd(2, 0.5, 2000,1);
n2 = normrnd(2, 0.1, 2000,1);

trndata = cat(2,n1,n2);

q1 = normrnd(3, 0.5, 1000,1);
q2 = normrnd(3, 0.5, 1000,1);

qualdata = cat(2,q1,q2);

[q,n] = gmm_test(trndata, qualdata,1,[1,6]);

%Example 2    
n11 = normrnd(2, 0.5, 2000,1);
n12 = normrnd(3, 0.1, 2000, 1);
n21 = normrnd(2, 0.1, 2000,1);
n22 = normrnd(3.5,0.5,2000,1)

n1 = cat(1,n11,n12)
n2 = cat(1,n21,n22)

trndata = cat(2,n1,n2);

q1 = normrnd(4, 0.7, 1000,1);
q2 = normrnd(4, 0.5, 1000,1);

qualdata = cat(2,q1,q2);

[q,n] = gmm_test(trndata, qualdata,1,[1,6]);


%% Load data and create train-test sets
data = readtable('hairdryer.csv');
X = table2array(data(:,1));
Y = table2array(data(:,2));
rng(1);
[train_idx, ~, test_idx] = dividerand(size(X,1), 0.8, 0,0.2);
X_train = X(train_idx,:);
X_test = X(test_idx,:);
Y_train = Y(train_idx,:);
Y_test = Y(test_idx,:);

%% Train initial Takagi-Sugeno model
opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 2;
ts_model = genfis(X_train,Y_train,opt);

%% Check initial performance on test set
Y_pred_initial = evalfis(ts_model, X_test);
rmse_initial = rmse(Y_pred_initial, Y_test);
fprintf('Initial RMSE: %4.3f \n', rmse_initial);

%% Tune initial model using ANFIS
[in,out,rule] = getTunableSettings(ts_model);
anfis_model = tunefis(ts_model,[in;out],X_train,Y_train,tunefisOptions("Method","anfis"));

%% Check ANFIS tuned model performance
Y_pred_final = evalfis(anfis_model, X_test);
rmse_final = rmse(Y_pred_final, Y_test);
fprintf('Final RMSE: %4.3f \n', rmse_final);
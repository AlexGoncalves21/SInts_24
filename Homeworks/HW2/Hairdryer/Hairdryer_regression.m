%% Load data and create train-test sets
data = readtable('hairdryer.csv');
X = table2array(data(:,1));  % Input feature
Y = table2array(data(:,2));  % Output/target variable
rng(1); 
[train_idx, ~, test_idx] = dividerand(size(X,1), 0.8, 0, 0.2);
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

mse_initial = mean((Y_pred_initial - Y_test).^2);

mape_initial = mean(abs((Y_test - Y_pred_initial) ./ Y_test)) * 100;

var_residual = var(Y_test - Y_pred_initial);
var_total = var(Y_test);
explained_variance_initial = 1 - var_residual / var_total;

fprintf('Initial MSE: %4.3f \n', mse_initial);
fprintf('Initial MAPE: %4.3f%% \n', mape_initial);
fprintf('Initial Explained Variance: %4.3f \n', explained_variance_initial);

%% Tune initial model using ANFIS
[in, out, rule] = getTunableSettings(ts_model);
anfis_model = tunefis(ts_model, [in; out], X_train, Y_train, tunefisOptions("Method","anfis"));

%% Check ANFIS tuned model performance
Y_pred_final = evalfis(anfis_model, X_test);

mse_final = mean((Y_pred_final - Y_test).^2);

mape_final = mean(abs((Y_test - Y_pred_final) ./ Y_test)) * 100;

var_residual_final = var(Y_test - Y_pred_final);
var_total_final = var(Y_test);
explained_variance_final = 1 - var_residual_final / var_total_final;

fprintf('Final MSE: %4.3f \n', mse_final);
fprintf('Final MAPE: %4.3f%% \n', mape_final);
fprintf('Final Explained Variance: %4.3f \n', explained_variance_final);

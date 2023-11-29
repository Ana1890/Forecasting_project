end_train_date = '2014-01-01'
start_test_date = '2014-02-01'

steps = 36

date_format = '%Y%m'

target_variable = 'Adjusted Passenger Count'
date_variable = 'Activity Period Datetime'
exogenous_variable = 'Price Category Code_Low Fare'

# Mlflow
experiment_name_set = 'Child Runs Fourth'

num_sets = 3

param_grid1 = {'n_estimators': [25, 50, 100],
                'max_depth': [4, 5, 6]}
lags_grid1 = [5, 12, 16]


param_grid2 = {'n_estimators': [400, 500],
                'max_depth': [2,3]}
lags_grid2 = [5, 20]


param_grid3 = {'n_estimators': [30, 60, 90],
               'max_depth': [2,4,8]}
lags_grid3 = [5,9,12]

# For child runs
test_ident = 'second'
num_runs = 5


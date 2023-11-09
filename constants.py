end_train_date = '2014-01-01'
start_test_date = '2014-02-01'

steps = 36

date_format = '%Y%m'

target_variable = 'Adjusted Passenger Count'
date_variable = 'Activity Period Datetime'
exogenous_variable = 'Price Category Code_Low Fare'

param_grid = {'n_estimators': [25, 50, 100, 200, 500],
                'max_depth': [2, 3, 5, 10, 15]}

lags_grid = [5, 12, 16, 20]

import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.models import infer_signature

from constants import *

def get_and_process_data():
    """Get the specific data to be trained.
    
    we dont need to pass any variable."""
    df_traffic = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')
    # print(list(df_traffic.columns))
    df_traffic['Activity Period Datetime'] = pd.to_datetime(df_traffic['Activity Period'], format=date_format)
    df_selected = df_traffic[['Activity Period Datetime','Adjusted Passenger Count', 'Price Category Code']]
    df_dumm = pd.get_dummies(df_selected, columns=['Price Category Code'], dtype='int64')

    df_grouped = df_dumm.groupby(date_variable)[[target_variable, exogenous_variable]].sum()
    y = df_grouped.resample('MS')[[target_variable, exogenous_variable]].mean()

    return y

def train_model(y: pd.DataFrame):
    """Train a model given a specific data."""
    data_train = y[:-steps]
    data_test  = y[-steps:]

    
    forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # This value will be replaced in the grid search
             )

    results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = data_train[target_variable],
                        exog        = data_train[exogenous_variable],
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = steps,
                        refit       = False,
                        metric      = 'mean_squared_error',
                        initial_train_size = int(len(data_train)*0.5),
                        return_best = True,
                        n_jobs      = 'auto',
                        verbose     = False
                )
    return forecaster, data_train, data_test, results_grid  

def main():
    """Main function."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Flights estimators")

    df_grouped = get_and_process_data()
    with mlflow.start_run(run_name="five_constants"):
        forecaster, data_train, data_test, results_grid = train_model(df_grouped)
        predictions = forecaster.predict(steps=steps, exog=data_test[exogenous_variable])

        error_mse = mean_squared_error(
                    y_true = data_test[target_variable],
                    y_pred = predictions
                )

        print(f"Test error (mse): {error_mse}")
        print(f"N estimators choosed: {results_grid.reset_index().loc[0,'n_estimators']}")
        mlflow.log_param("n_estimators", results_grid.reset_index().loc[0,'n_estimators'])
        mlflow.log_param("max_depth", results_grid.reset_index().loc[0,'max_depth'])
        mlflow.log_metric("test_rmse", error_mse)

        predictions_t = forecaster.predict(steps=steps, exog=data_test[exogenous_variable])
        signature = infer_signature(data_train, predictions_t)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                forecaster, "model", registered_model_name="forecaster_model", signature=signature
            )
        else:
            mlflow.sklearn.log_model(forecaster, "model", signature=signature)



    # return results

if __name__  == "__main__":
    main()
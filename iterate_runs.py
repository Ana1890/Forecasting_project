"""What if we need to run this again??."""
import pandas as pd
import numpy as np
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature

from random import sample
from functools import partial
from itertools import starmap
from more_itertools import consume

from constants import *

def plot_predictions(data_train, data_test, predictions, save_path):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    data_train['Adjusted Passenger Count'].plot(ax=ax, label='train')
    data_test['Adjusted Passenger Count'].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)

    return fig


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

def train_model(y: pd.DataFrame, params_grid: list, lags_grid: list):
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
                        param_grid  = params_grid,
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




def execute_and_log(run_name, df_grouped, params_grid, lags_grid, test_no, tag_ident):
    with mlflow.start_run(run_name=run_name, nested=True):

        nro = int(run_name.split('_')[1])
        # Train model
        forecaster, data_train, data_test, results_grid = train_model(df_grouped, params_grid[nro], lags_grid[nro])
        predictions = forecaster.predict(steps=steps, exog=data_test[exogenous_variable])

        # Log metrics
        error_mse = mean_squared_error(
                    y_true = data_test[target_variable],
                    y_pred = predictions
                )

        print(f"Test error (mse): {error_mse}")
        print(f"N estimators choosed: {results_grid.reset_index().loc[0,'n_estimators']}")
        mlflow.log_param("n_estimators", results_grid.reset_index().loc[0,'n_estimators'])
        mlflow.log_param("max_depth", results_grid.reset_index().loc[0,'max_depth'])
        mlflow.log_metric("test_rmse", error_mse)

        # Predict uknown data
        predictions_t = forecaster.predict(steps=steps, exog=data_test[exogenous_variable])

        # Log artifacts
        predictions_plot = plot_predictions(data_train, data_test, predictions_t, save_path=f"images_plot/{tag_ident}_{nro}_predictions_plot.png")
        mlflow.log_artifact(f'images_plot/{tag_ident}_{nro}_predictions_plot.png')
        signature = infer_signature(data_train, predictions_t)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Register the model
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(
                forecaster, "model", registered_model_name=f"forecaster_model_{tag_ident}_{nro}_{test_no}", signature=signature
            )
        else:
            mlflow.sklearn.log_model(forecaster, "model", signature=signature)




def generate_run_names(ident, params_grid):
    return (f"run_{i}_{ident}_test" for i in range(len(params_grid)))




def tunning_process(test_no, df_grouped, params_grid, lags_grid, test_identifier):

    ident = "default" if not test_identifier else test_identifier

    with mlflow.start_run(run_name=f"parent_run_test_{ident}_{test_no}"):

        log_current_run = partial(execute_and_log, 
                                  df_grouped=df_grouped, 
                                  params_grid=params_grid, 
                                  lags_grid=lags_grid,
                                  test_no=test_no,
                                  tag_ident=ident                                                                    
                                  )

        mlflow.set_tag("test_identifier", ident)
        # Generate run names and apply log_current_run function to each run name
        runs = starmap(log_current_run,  ((run_name,) for run_name in generate_run_names(ident, params_grid)))
        # Consume the iterator to execute the runs
        consume(runs)
        
        



def main():
    """Main function."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name_set)

    df_grouped = get_and_process_data()

    params_grid = [param_grid1, param_grid2, param_grid3]
    lags_grid = [lags_grid1, lags_grid2, lags_grid3]

    consume(starmap(tunning_process, ((x, df_grouped, sample(params_grid, len(params_grid)), sample(lags_grid, len(lags_grid)), test_ident) for x in range(num_runs))))
    

if __name__  == "__main__":
    main()
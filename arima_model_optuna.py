import optuna
import mlflow

import statsmodels.api as sm
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse

from itertools import starmap
from more_itertools import consume
from constants_arima import *

def get_and_process_data():
    """Get the specific data to be trained"""
    df_traffic = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')
    format='%Y%m'
    df_traffic['Activity Period Datetime'] = pd.to_datetime(df_traffic['Activity Period'], format=format)
    df_grouped = df_traffic.groupby('Activity Period Datetime')['Adjusted Passenger Count'].sum()
    return df_grouped

def objective(trial):
    with mlflow.start_run():
        df_data = get_and_process_data()
        y = df_data.resample('MS').mean()

        y_train = y[:'2014-01-01']
        y_test = y['2014-02-01':]


        p = trial.suggest_categorical('p', [0, 1, 2])
        d = trial.suggest_categorical('d', [0, 1, 2])
        q = trial.suggest_categorical('q', [0, 1, 2])

        mod = sm.tsa.statespace.SARIMAX(y_train,
                                        order=(p, d, q),
                                        seasonal_order=(p, d, q, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()

        aic = results.aic

        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)

        mlflow.log_metric("aic", aic)


        return aic



def main():
    """Main function."""

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    # Log the best parameters
    mlflow.log_params(study.best_params)
    mlflow.log_metric('best_score', study.best_value)


if __name__  == "__main__":
    main()
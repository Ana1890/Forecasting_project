import mlflow
from random import sample
import optuna
import pandas as pd
import statsmodels.api as sm

from constants_arima import *

pdq1 = [[1,0],
        [1,0,2],
        [1,0]]


pdq2 = [[1,0,2],
        [1,0],
        [0,1]]

pdq3 = [[1,0],
        [1,0],
        [1,0,2]]


def get_and_process_data():
    """Get, parse, and create stationary data."""

    df_traffic = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')
    format='%Y%m'
    df_traffic['Activity Period Datetime'] = pd.to_datetime(df_traffic['Activity Period'], format=format)
    df_grouped = df_traffic.groupby('Activity Period Datetime')['Adjusted Passenger Count'].sum()
    return df_grouped


def sweep_model(trial, pdq):
    """Objective model to be optimized into a parent run."""
    # Start a child run
    with mlflow.start_run(run_name=f'ChildRun_{trial}', nested=True) as child_run:
        df_data = get_and_process_data()
        y = df_data.resample('MS').mean()

        y_train = y[:'2014-01-01']
        y_test = y['2014-02-01':]


        p = trial.suggest_categorical('p', pdq[0])
        d = trial.suggest_categorical('d', pdq[1])
        q = trial.suggest_categorical('q', pdq[2])

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
    """Set the inisialization of an experiment."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    pdq_join = [pdq1, pdq2, pdq3]

    for i in range(len(pdq_join)):
        pdq = pdq_join[i]

        total_shape = [len(pdq[0]),len(pdq[1]),len(pdq[2])]

        # Start a parent run
        with mlflow.start_run(run_name=f'ParentRun_{total_shape}') as parent_run:
            # Log parameters or tags specific to the parent run
            mlflow.log_param('hyperparams_shape', total_shape)

            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: sweep_model(trial, pdq), n_trials=20)



if __name__  == "__main__":
    main()
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

def train_model(test_no, p,d,q):
    """Train a model given a specific data."""
    with mlflow.start_run(run_name=f"{str(test_no + 1)}_variable"):
        df_data = get_and_process_data()
        y = df_data.resample('MS').mean()

        y_train = y[:'2014-01-01']
        y_test = y['2014-02-01':]

        pp, dp, qp = p[test_no], d[test_no], q[test_no]
        mod = sm.tsa.statespace.SARIMAX(y_train,
                                        order=(pp, dp, qp),
                                        seasonal_order=(pp, dp, qp, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()

        aic = results.aic

        mlflow.log_param("p", pp)
        mlflow.log_param("d", dp)
        mlflow.log_param("q", qp)

        mlflow.log_metric("aic", aic)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Register the model
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(
                results, "model", registered_model_name=f"arima_model_{str(test_no+1)}"
            )
        else:
            mlflow.sklearn.log_model(results, "model")


def main():
    """Main function."""

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    consume(starmap(train_model, ((x, p, d, q) for x in range(3))))


if __name__  == "__main__":
    main()
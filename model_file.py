import statsmodels.api as sm
import pandas as pd

def get_and_process_data():
    """Get the specific data to be trained"""
    df_traffic = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')
    format='%Y%m'
    df_traffic['Activity Period Datetime'] = pd.to_datetime(df_traffic['Activity Period'], format=format)
    df_grouped = df_traffic.groupby('Activity Period Datetime')['Adjusted Passenger Count'].sum()
    return df_grouped

def train_model(df_data: pd.DataFrame):
    """Train a model given a specific data."""
    y = df_data.resample('MS').mean()

    y_train = y[:'2014-01-01']
    y_test = y['2014-02-01':]


    mod = sm.tsa.statespace.SARIMAX(y_train,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return results

def main():
    """Main function."""
    df_grouped = get_and_process_data()
    results = train_model(df_grouped)
    print(results.summary().tables[1])
    # return results

if __name__  == "__main__":
    main()
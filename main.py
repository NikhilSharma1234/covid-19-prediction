# Authors:
#   Amber Hankins
#   Nikhil Sharma
#   CS 431 - Big Data
#   Topic: Covid-19 Prediction

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import glob


state_names = ["California"]


def main(state):
    # Convert first X days of 2021 into individual dataframes for training data
    df_train = pd.DataFrame()
    all_files = glob.glob('data/*.csv')
    for file in all_files:
        df_train = pd.concat([df_train, fetchData(state, file)])
    # Convert next day of 2021 into individual dataframes for testing data
    df_test = fetchData(state, '03-29-2021.csv')

    # Output State Data to a csv
    df_train.to_csv(state + '_data.csv')

    # Get training and testing info
    X_train = df_train.drop(columns=['Confirmed'])
    X_test = df_test.drop(columns=['Confirmed'])
    y_train = df_train['Confirmed']
    y_test = df_test['Confirmed']

    # Linear Regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    print("State: " + state)
    print("Linear Regression Confirmed Cases Prediction: " + str(preds))
    dist = np.linalg.norm(preds-y_test)
    print("Linear Regression Prediction Precision: " + str(dist))

def fetchHeaders(url):
    df = pd.read_csv(url, index_col=0)
    df = df.iloc[[0]]


def fetchData(state, url):
    # Read csv into pandas dataframe
    df = pd.read_csv(url, index_col=0)

    # Drop redundant and non-numerical rows
    df = df.drop(columns=['Lat', 'Long_', 'UID', 'FIPS', 'Country_Region', 'Last_Update', 'ISO3', 'Date', 'Recovered', 'Active', 'People_Hospitalized', 'Hospitalization_Rate', 'People_Tested', 'Mortality_Rate'])

    # Get Alaska only
    try:
        df = df.loc[[state]]
    except:
        df = pd.DataFrame()

    # Return dataframe
    return df

if __name__ == '__main__':
    for state in state_names:
        main(state)  
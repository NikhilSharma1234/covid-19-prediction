# Authors:
#   Amber Hankins
#   Nikhil Sharma
#   CS 431 - Big Data
#   Topic: Covid-19 Prediction

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    # Convert first five days of 2021 into individual dataframes for training data
    df_train = pd.DataFrame()
    df_train = pd.concat([df_train, fetchData('01-01-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-02-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-03-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-04-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-05-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-06-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-07-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-08-2021.txt')], ignore_index=True)
    df_train = pd.concat([df_train, fetchData('01-09-2021.txt')], ignore_index=True)


    # Convert next five days of 2021 into individual dataframes for testing data
    df_test = fetchData('01-10-2021.txt')

    X_train = df_train.drop(columns=['Confirmed'])
    X_test = df_test.drop(columns=['Confirmed'])

    y_train = df_train['Confirmed']
    y_test = df_test['Confirmed']

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    dist = np.linalg.norm(preds-y_test)
    print("Linear Regression Prediction Precision: " + str(dist))
    

def fetchData(url):
    # Read csv into pandas dataframe
    df = pd.read_csv(url, index_col=0)

    # Drop all columns that contain only NaNs
    df = df.dropna(axis=1, how='all')

    # Dropping rows that still contain NaNs (Cruise Ships and American Samoa)
    df = df.dropna(axis=0)

    # Drop redundant and non-numerical rows
    df = df.drop(columns=['Country_Region', 'Last_Update', 'ISO3', 'Date'])

    # Return dataframe
    return df

if __name__ == '__main__':
    main()
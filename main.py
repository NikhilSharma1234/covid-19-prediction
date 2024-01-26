# Authors:
#   Amber Hankins
#   Nikhil Sharma
#   CS 431 - Big Data
#   Topic: Covid-19 Prediction

import pandas as pd

def main():
    print('Hello world!')
    fetchData()

def fetchData():
    url = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_daily_reports/01-01-2021.csv'
    df = pd.read_csv(url, index_col=0)
    print(df.head(5))

if __name__ == '__main__':
    main()
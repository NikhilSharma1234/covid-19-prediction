# Authors:
#   Amber Hankins
#   Nikhil Sharma
#   CS 431 - Big Data
#   Topic: Covid-19 Prediction

import pandas as pd
import io
import requests
import glob
from dotenv import load_dotenv
import os

load_dotenv()
def main():
    print('Hello world!')
    fetchData()

def fetchData():
    path = os.getenv('CSVPATH')
    all_files = glob.glob(path)
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df[df['Province_State'] == 'Alabama'])
    frame = pd.concat(li, axis=0, ignore_index=True)
    print(frame.to_string())

if __name__ == '__main__':
    main()
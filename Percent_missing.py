import pandas as pd
import numpy as np

def percent_missing(path):
    data = pd.read_csv(path)
    #deta = pd.read_csv("Data/merged_data_tim_run1.csv")
    #print(data.shape, deta.shape)
    missing = data.isnull().sum()
    #print("missing values:", missing)
    percent_missing = data.isnull().sum() * 100 / len(data)
    print("percentage missing:", percent_missing)

#percent_missing('Full_data/full_dataset_tim_run_1.csv')
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

#percent_missing('Data/Full_data/full_orig_data/full_dataset_viv_run_1.csv')

def remove_first_2_columns(path):
    df = pd.read_csv(path)
    df.drop(df.columns[df.columns.str.contains('^Unnamed')], axis=1, inplace=True)
    df.to_csv(path, index=False)

remove_first_2_columns('Data/Full_data/hp_data/highpass_vivian_run_2.csv')

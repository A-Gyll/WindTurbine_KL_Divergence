import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os
import math

def read_in_data(file_path):
    """
    read in data
    """
    df = pd.read_csv(file_path,  delimiter=';')
    return df

def collect_normal_operation_data(data_filepath):
    csv_files = [file for file in os.listdir(data_filepath) if file.endswith('.csv')]

    filtered_dataframes = []

    # Loop through each file and filter through normal data to make one dataframe
    for file_name in csv_files:
        file_path = os.path.join(data_filepath, file_name)
        df = pd.read_csv(file_path, delimiter=';')
        filtered_df = df[df['status_type_id'] == 0]
        filtered_dataframes.append(filtered_df)
    
    # Concatenate all filtered DataFrames
    filtered_df = pd.concat(filtered_dataframes, ignore_index=True)
    
    total_rows = len(filtered_df)

    # If more than 80% of sensor values are 0, then remove
    threshold = total_rows * 0.8
    columns_to_drop = [col for col in filtered_df.columns if (filtered_df[col] == 0).sum() > threshold]
    filtered_df = filtered_df.drop(columns=columns_to_drop)

    return filtered_df


if __name__ == "__main__": 
    read_in_data("/home/ujx4ab/ondemand/data/WT_Data/WF_A/datasets")

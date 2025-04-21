import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from {filepath} : {e}")
#df = pd.read_csv('https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv')

def load_params(filepath : str) -> float:
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"Error in loading params from {filepath} : {e}")
#test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

# X = df.iloc[:,0:-1]
# y = df.iloc[:,-1]
def split_data(data: pd.DataFrame, test_size:float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data,test_size = test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error in splitting data train_test_split : {e}")
#train_data, test_data = train_test_split(df,test_size=test_size,random_state=42)

# data_path = os.path.join("df","raw")

# os.makedirs(data_path)

# train_data.to_csv(os.path.join(data_path, "train.csv"),index=False)
# test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)
def save_data(df:pd.DataFrame, filepath:str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error in saving data to {filepath} : {e}")

def main():
    data_filepath = "https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("df","raw")

    df = load_data(data_filepath)
    test_size = load_params(params_filepath)
    train_data, test_data = split_data(df,test_size)

    os.makedirs(raw_data_path)

    save_data(train_data,os.path.join(raw_data_path,"train.csv"))
    save_data(test_data,os.path.join(raw_data_path,"test.csv"))


if __name__ == "__main__":
    main()
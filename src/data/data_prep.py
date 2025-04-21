import numpy as np
import pandas as pd
import os

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from {filepath} : {e}")
# train_data = pd.read_csv('./df/raw/train.csv')
# test_data = pd.read_csv('./df/raw/test.csv')

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error in filling missing values with median : {e}")


# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

def save_file(df : pd.DataFrame, filepath : str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error in saving file to {filepath}: {e}")


def main():
    try:
        train_raw_data_path = "./df/raw/train.csv"
        test_raw_data_path = "./df/raw/test.csv"
        #processed_data_path = "./df/processed"

        train_data = load_data(train_raw_data_path)
        test_data = load_data(test_raw_data_path)

        train_processed = fill_missing_with_median(train_data)
        test_processed = fill_missing_with_median(test_data)

        processed_data_path = os.path.join("df","processed")
        os.makedirs(processed_data_path)

        save_file(train_processed,os.path.join(processed_data_path,"train_processed.csv"))
        save_file(test_processed, os.path.join(processed_data_path,"test_processed.csv"))
    
    except Exception as e:
        raise Exception(f"Error occurred : {e}")




if __name__ == "__main__":
    main()


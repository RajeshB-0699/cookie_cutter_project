import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn
import pickle
import os
import joblib
import yaml




def load_params(params_path :str) -> int:
    try:
        with open(params_path,"r") as file:
            params = yaml.safe_load(file)
            return params['model_building']['n_estimators']
    except Exception as e:
        raise Exception(f"Error in loading params from {params_path} : {e}")
    
def load_data(filepath : str)-> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data file from {filepath} : {e}")
    
def prepare_data(data : pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns = ['Potability'], axis = 1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error in preparing dataset  : {e}")
        
        
# n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]

# X_train = train_processed.iloc[:,0:-1].values
# y_train = train_processed.iloc[:,-1].values
# X_test = test_processed.iloc[:,0:-1].values
# y_test = test_processed.iloc[:,-1].values

# X_train = train_processed.drop(columns=['Potability'],axis=1)
# y_train = train_processed['Potability']

def train_model(X : pd.DataFrame, y: pd.Series, n_estimators : int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X,y)
        return clf
    
    except Exception as e:
        raise Exception(f"Error in training model with estimators {n_estimators} : {e}")
    

def save_model(model : RandomForestClassifier,filepath : str) -> None:
    try:
        with open(filepath,"wb") as file:
            pickle.dump(model,file)
    except Exception as e:
        raise Exception(f"Error in saving model to filepath  {filepath} : {e}")

#pickle.dump(clf,open("model.pkl","wb"))

def main():
    try:
        path_params = "params.yaml"
        train_processed_path  = './data/processed/train_processed.csv'
    #    test_processed_path  = './df/processed/test_processed.csv'
        model_name = "models/model.pkl"

        estimators = load_params(path_params)
        df = load_data(train_processed_path)
        X_train, y_train = prepare_data(df)
        model = train_model(X_train, y_train,estimators)
        save_model(model,model_name)
    except Exception as e:
        raise Exception(f"Error in building model : {e}")

if __name__ == "__main__":
    main()






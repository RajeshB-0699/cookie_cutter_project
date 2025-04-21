import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score



def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading file from {filepath} : {e}")
#test_processed = pd.read_csv('./df/processed/test_processed.csv')



def prepare_data(data : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns = ['Potability'], axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error in preparing data test dataset : {e}")
# X_test = test_processed.iloc[:,0:-1].values
# y_test = test_processed.iloc[:,-1].values



def load_model(filepath : str):
    try:
        with open(filepath,'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error in loading model from {filepath}: {e}")  
#model = pickle.load(open('model.pkl','rb'))



def evaluation_model(model, X_test: pd.DataFrame, y_test : pd.Series) -> dict:

    try:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1,
            'recall_score': recall
        }
        
        return metrics_dict
    
    except Exception as e:
        raise Exception(f"Error in evaluating model {e}")
    
def save_metrics(metrics_dict : dict, filepath : str) -> None:
    try:
        with open('metrics.json','w') as file:
            json.dump(metrics_dict,file,indent=4)
    except Exception as e:
        raise Exception(f"Error in saving metics in {filepath} : {e}")
    

def main():
    try:
        test_data_path = './df/processed/test_processed.csv'
        model_name = 'model.pkl'
        metrics_filepath = 'metrics.json'


        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_name)
        evaluation_dict = evaluation_model(model,X_test,y_test)
        save_metrics(evaluation_dict,metrics_filepath)
    except Exception as e:
        raise Exception(f"Error in evaludating mode : {e}")

if __name__ == '__main__':
    main()




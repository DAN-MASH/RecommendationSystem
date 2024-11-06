import os
import sys
import pandas as pd
import numpy as np 

import pickle
import dill

from src.exception import CustomException
from src.components import model_trainer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        #make sure the directory exists
        dir_path=os.path.dirname(file_path)
        dir_path=os.makedirs(dir_path,exist_ok=True)

        #save the file obj
        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluation_models(X_train,y_train,X_test,y_test,models,params):
    try:
        #create reprt to hold the predicted values of each model
        report={}
        #iterate through the models dictionary as a list
        for i in range(len(list(models))):
            #get the models from the models value
            model=list(models.values())[i]
            param=param[list(models.key())[i]]
            
            gs=GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            #evaluation
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            #append the keys of each model to the report
            report[list(model.keys())[i]]=test_model_score
            return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open (file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

        
    
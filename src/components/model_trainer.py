import pandas as pd
import numpy as np

import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluation_models
from src.utils import save_object

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts',"model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_trainer(self,train_array,test_array):
        try:
            #step 1: Split the data into X_train,y_train, X_test and y_test
            logging.info("Splitting the data into train and test")
            X_train,y_train=train_array[:,:-1],train_array[:,-1]
            X_test,y_test=test_array[:,:-1],test_array[:,-1]
            #step 2: call the models and contain them in a dictonary
            models={
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "Linear Regression":LinearRegression(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "Cat Boost Regressor":CatBoostRegressor(),
                "XGB Regressor":XGBRegressor()
            }
            #step 3: set up the hyperparameter for model tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            #step 4: Make prediction by calling the evaluation model from utils.
            model_report:dict=evaluation_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            logging.info(f"Model report:{model_report}")
            #get the best model based in r2_score
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.key())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=best_model_name

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and test datasets")
            #step 5: train the data on the best model
            best_model.fit(X_train,y_train)
            #step6: save the model
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            #step 7: evaluate the model
            y_pred=best_model.predict(X_test)
            r2_score=r2_score(y_test,y_pred)
            return r2_score
        except Exception as e:
            raise CustomException(e,sys)

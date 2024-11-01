import os
import numpy as np
import pandas as pd
import sys

from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass


@dataclass
class PredictionPipeline:
    def __int__(self):
        pass
    def predict(self,features):
        try:
            #step 1:read the model and preprocessor pkl folders
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")
            print("Before loading")
            #step 2: load the model and preprocessor
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After loading")
            #step 3: scale the data
            scaled_data=preprocessor.transform(features)
            #make prediction
            pred=model.predict(scaled_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
#reponsible for mapping the input in html to the backend
class CustomData:
    def __init__(self,):
        pass
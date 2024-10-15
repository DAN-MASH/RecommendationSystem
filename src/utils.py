import os
import sys
import pandas as pd
import numpy as np 

import pickle
import dill

from src.exception import CustomException


def save_object(file_path,obj):
    try:
        #make sure the directory exists
        dir_path=os.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open (file_path ,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
    
    
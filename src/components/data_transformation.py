import pandas as pd
import numpy as np
import os

import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


@dataclass
class DataTransformationConfig:
    #save the tranformation model so it can be loaded later during training and prediction
    preprocessor_obj_file_path:str=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DataTranformation_Config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            #step 1: Define numerical and categorical data
           numerical_cols= ['math score', 'reading score', 'writing score']
           categorical_cols= ['gender', 'race/ethnicity', 'parental level of education', 
                                'lunch', 'test preparation course']
           #step 2: Defince the steps in the pipeline
           num_pipeline=Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy='median')),
                   ("scaler", StandardScaler())
               ]
           )

           cat_pipeline=Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy="most_frequent")),
                   ("encoder",OneHotEncoder()),
                   ("scaler", StandardScaler())
               ]
           )
           logging.info(f"numerical_columns: {numerical_cols}")
           logging.info(f"categorical_columns: {categorical_cols}")

           #step 3: combine the num and cat pipeline
           preprocessor=ColumnTransformer(
               [
               ("num_pipeline",num_pipeline,numerical_cols),
               ("cat_pipeline",cat_pipeline,categorical_cols)
               ]
           )
           logging.info("Data transformer activated successfully")
           return preprocessor
    
        except Exception as e:
            logging.info("Error in creating data transformer object:")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            #read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read and train data completed")
            logging.info(f"Train_dataframe shape: {train_df}, Test_dataframe shape: {test_df}")

            #Extract features and target variables
            target_column='average_score'
            input_train_features=train_df.drop(columns=target_column,axis=1)
            target_train=target_column

            input_test_features=test_df.drop(columns=target_column,axis=1)
            target_test=target_column

            #step3: get the preprocessor obj and transform the data 
            logging.info("start the data transformation process")
            preprocessor_obj=self.get_data_transformation_object()
            input_train_features_transformed=preprocessor_obj.fit_transform(input_train_features)
            input_test_features_transformed=preprocessor_obj.transform(input_test_features)
            logging.info("Data transformation completed")
            #combine the feature and target data
            train_arr=np.c_[
                input_train_features_transformed,np.array(target_train)
            ]
            test_arr=np.c_[
                input_test_features_transformed,np.array(target_test)
            ]
            #save the preprocessor obj for fure use
            save_object(file_path=self.DataTranformation_Config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            return (
                train_arr,
                test_arr,
                self.DataTranformation_Config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error in the data transformation process")
            raise CustomException(e,sys)
        
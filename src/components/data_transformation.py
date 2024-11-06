import pandas as pd
import numpy as np
import os

import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    transformation_obj:str=os.path.join('artifacts','preprocessor.pkl')
    transformed_data_path:str=os.path.join('artifacts', "transformed_data.csv")

class DataTransformation:
    def __init__(self):
        
        self.data_transformation_config=DataTransformationConfig()

    def data_transformation(self, df):
        try:
            #remove null
            df=df.dropna()
            logging.info(f"Null values dropped")
            #remove duplicates
            df.drop_duplicates(inplace=True)
            logging.info("Duplicates removed")

            #normalize the text
            df['title']=df['title'].str.lower().str.strip()
            df['body'] = df['body'].str.lower().str.strip()
            df['tags'] = df['tags'].str.lower().str.replace(" ","")
            df['status']=df['status'].str.lower().str.strip()
            logging.info("Text normalization completed for 'title', 'body', and 'tags'")
            # Ensure numerical columns are of numeric type
            df['likes']=df['likes'].astype('int64')
            df['views']=df['views'].astype('int64')
            logging.info(f"Numerical data confirmed: {df.select_dtypes(include=['int64','float64'])}")
            #reset index to the original after conversion and normalization
            df.reset_index(drop=True,inplace=True)
            return df
        except Exception as e:
            logging.info(f"Error occured during data transformation: {e}")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path):
        try:
            #train_data_path='artifacts\train_data.csv'
            train_data=pd.read_csv(train_path)
            #get the data_tranformation instance
            tranformed_data=self.data_transformation(train_data)
            logging.info(f"Data Tranformation intiated. Tranformed data shape: {tranformed_data.shape}")
            #make sure the directory exists
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path),exist_ok=True)
            #save the transformed data to csv
            tranformed_data.to_csv(self.data_transformation_config.transformed_data_path,index=False)
            logging.info(f"Transformed data saved at {self.data_transformation_config.transformed_data_path}")
            save_object(file_path=self.data_transformation_config.transformation_obj,
                        obj=tranformed_data)
            return (tranformed_data,
                    self.data_transformation_config.transformation_obj,
                    self.data_transformation_config.transformed_data_path)

        except Exception as e:
            logging.info("Error occured during data initiation")
            raise CustomException(e, sys)
    
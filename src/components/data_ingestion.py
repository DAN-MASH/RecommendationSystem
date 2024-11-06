import sys
import pandas as pd
import os
import requests 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluation_models
#from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


#decorator

@dataclass
class DataIngestionConfig:
    #artifacts to store the data and the datapaths as strings
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')
    train_data_path:str=os.path.join('artifacts','train_data.csv')

#Data ingestion class
class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()
    def fetch_data(self):
        try:
            url="https://techtales.vercel.app/api/blogs"
            response=requests.get(url)
            df=response.json()
            #convert the json data to dataframe
            df=pd.json_normalize(df)
            return df

        except Exception as e:
            logging.info("Failed to get data from API: {e}")
            raise CustomException(e,sys)
       
    def initiate_data_ingestion(self):
        try:
            logging.info("Start data ingestion")
            #stpe 1: Fetch the data from the API
            df=self.fetch_data()
            #step 2: converrt the data into a dataframe
            #data=pd.DataFrame(data)
            #df = pd.read_json(self.config.raw_data_path)
            #make sure the directort exists
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            #save the data as csv
            df.to_csv(self.config.raw_data_path,index=False)
            logging.info("Data ingestion completed.{\n} Data shape: {df.shape}")
            #get the data used for training by getting only the necessary columns
            #make sure the directort exists
            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)
            columns=['id', 'title', 'body', 'slug', 'tags','status', 'likes',
                    'views','image.url']
            train_data=df[columns]
            #save to csv
            train_data.to_csv(self.config.train_data_path,index=False)
            logging.info(f"Train data saved at {self.config.train_data_path}. Train_data saved at: {self.config.train_data_path}")
            return (self.config.raw_data_path,
                    self.config.train_data_path)
        except Exception as e:
            logging.info("Error occured during data ingestion")
            raise CustomException(e,sys)


if __name__=="__main__":
    ingestion=DataIngestion()
    raw_data,train_data=ingestion.initiate_data_ingestion()
    logging.info(f"Data ingestion completed.Data at: {raw_data}.Train data at:{train_data}")


    data_tranformation=DataTransformation()
    transformed_data,transformation_obj,transformed_data_path=data_tranformation.initiate_data_transformation(train_data)
    logging.info(f"Data transformation completed. Transformed data saved at: {transformed_data}. Transformation object saved at: {transformation_obj}")

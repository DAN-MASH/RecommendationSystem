import sys
import pandas as pd
import os
import requests 

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig

#decorator

@dataclass
class DataIngestionConfig:
    #artifacts to store the data and the datapaths as strings
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')

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
            data=pd.json_normalize(df)

        except Exception as e:
            logging.info("Failed to get data from API: {e}")
            raise CustomException(e,sys)
       
    def initiate_data_ingestion(self):
        try:
            logging.info("Start data ingestion")
            #stpe 1: Fetch the data from the API
            data=self.fetch_data()
            #step 2: converrt the data into a dataframe
            #data=pd.DataFrame(data)
            df = pd.read_json(self.config.raw_data_path)
            #make sure the directort exists
            os.makedirs(os.path.dirname(self.raw_data_path),exist_ok=True)
            #save the data as csv
            df.to_csv(self.config.raw_data_path,index=False)
            logging.info("Data ingestion completed.{\n} Data shape: {df.shape}")
            return self.config.raw_data_path
        except Exception as e:
            logging.info("Error occured during data ingestion")
            raise CustomException(e,sys)


if __name__=="__main__":
    ingestion=DataIngestionConfig()
    raw_data=ingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed.Data at: {raw_data}")
               


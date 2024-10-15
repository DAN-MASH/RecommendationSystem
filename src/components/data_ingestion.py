import sys
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#decorator
@dataclass
class DataIngestionConfig():
    #artifact folder to store the data
    #proived data ngestion with information on where to store the data
    raw_data_path: str = os.path.join('artifacts','raw_data.csv')
    train_data_path:str=os.path.join('artifacts','train_data.csv')
    test_data_path:str=os.path.join('artifacts','test_data.csv')
    

class DataIngestion():
   def __init__self(self):
       self.config=DataIngestionConfig()
       
       def initiate_data_ingestion(self):
           logging.info("Start the data ingestion")
           try:
               #Step1: read the raw data
               logging.info("reading raw data")
               df=pd.read(self.config.raw_data_path)
               logging.info(f"Raw data shape: {df.shape}")
               #save raw data to csv
               df.to_csv(self.config.raw_data_path,index=False)
               
               #step 2: Split the data to train and test set
               logging.info("Initiate Train test split")
               train_set, test_set=train_test_split(df, test_size=0.2,random_state=42)
               logging.info(f"train_set shape: {train_set.shape}")
               logging.info(f"test_set shape: {test_set.shape}")
               #step 3 save the train and test data as csv
               #make the train data path directort
               os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
               train_set.to_csv(self.config.train_data_path, index=False)
               test_set.to_csv(self.config.test_data_path, index=False)
               logging.info(f"Train and test sets saved at {self.config.train_data_path} and {self.config.test_data_path}")
               return self.config.train_data_path, self.config.test_data_path
           
           except Exception as e:
               logging.info(f"Error occured during ingestion: {e}")
               raise CustomException(e, sys)
           
if __name__=="__main__":
    ingestion=DataIngestion()
    train_data, test_data=ingestion.initiate_data_ingestion()
    logging.info(f"Data ingestion completed. Train data at: {train_data}, Test data at: {test_data}")
               


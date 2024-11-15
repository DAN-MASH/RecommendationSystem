import pandas as pd
import numpy as np

import os
import sys

from src.exception import CustomException
from src.logger import logging
#from src.utils import evaluation_models
from src.utils import save_object

from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts',"recommendation_model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def calculate_cosine_similarity(self,transformed_data):
        try:
            #step 1: TF-IDF Vectorizer
            tfidf=TfidfVectorizer(stop_words="english")
            #step 2: get the tfidf_matrix
            tfidf_matrix=tfidf.fit_transform(transformed_data['tags'])
            #step 3: calculate the cosine similarity
            cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
            return cosine_sim
        except Exception as e:
            raise CustomException(e,sys)
    def get_content_recommendation(self,blog_id,transformed_data,top_n):
        try:
            #get the cosine_similarity
            cosine_sim = self.calculate_cosine_similarity(transformed_data)

            #step 1: get the blog index that matches the blog id
            idx=transformed_data.index[transformed_data['id']==blog_id].tolist()[0]
            #step 2: Get the similarity score
            sim_score=list(enumerate(cosine_sim[idx]))
            #step 3:Sort the similarity score excluding the blog itself
            sim_score=sorted(sim_score,key=lambda x:x[1],reverse=True)[1:6]
            #step 4: Get the indices of the most similar blogs
            top_indices=[i[0] for i in sim_score]
            top_title=transformed_data.iloc[top_indices][['id','tags']].to_dict(orient="records")
            logging.info(f"Top {top_n} recommendations for Blog ID {blog_id}: {top_title}")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=cosine_sim)
            return (top_indices,
                    top_title,
                    self.model_trainer_config.trained_model_file_path)
        except Exception as e:
            logging.info("Error occured during content recommendation")
            raise CustomException(e,sys)
    


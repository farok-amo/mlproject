import os
import sys

from model_trainer import ModelTrainer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from exception import CustomException
from logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifects', "train.csv")
    test_data_path: str=os.path.join('artifects', "test.csv")
    raw_data_path: str=os.path.join('artifects', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("entered data ingestinon method")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("data imported")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=172)
            logging.info("train test split initiated")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion complete")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformer = DataTransformation()
    train_arr, test_arr,_ = data_transformer.initiate_data_transformation(
        train_path=train_path, test_path=test_path)

    print(ModelTrainer().initiate_model_trainer(train_arr, test_arr))



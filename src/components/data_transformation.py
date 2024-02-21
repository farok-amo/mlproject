import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from exception import CustomException
from logger import logging
from utils import save_object

   
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifects', "preprocessor.joblib")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(slef):
        '''
        function to transform data

        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education",
                                   "lunch", "test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())
                       ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding is complete")
            logging.info("Numerical columns scaling is complete")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipepline", num_pipeline, numerical_columns),
                    ("cat_pipepline", categorical_pipeline, categorical_columns),

                ]
            )

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data is completed")

            logging.info("obtaining preprocessor")
            preprocessor_obj = self.get_data_transformer_object()

            target_column="math_score"

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessor on training and testing DFs")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessor_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
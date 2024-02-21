import sys, os
import pandas as pd 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from logger import logging
from exception import CustomException
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("predict enterd")
            model = load_object('artifects\model.joblib')
            preprocessor = load_object('artifects\preprocessor.joblib')
            logging.info("model&preprocessor imported")
            data = preprocessor.transform(features)
            logging.info("preprocessor done")
            preds = model.predict(data)
            logging.info("model done")

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, gender:str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_frame(self):
        try:
            custom_data_input = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }
            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise CustomException(e, sys)
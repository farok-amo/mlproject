import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifects', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
          logging.info("splitting train and test data")
          X_train, y_train, X_test, y_test= (train_array[:,:-1], train_array[:,-1],
                                             test_array[:,:-1], test_array[:,-1]) 
          
          models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
          
          model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models) 

          best_model_score = max(sorted(model_report.values()))

          best_model_name = list(model_report.keys())[
              list(model_report.values()).index(best_model_score)]
          
          best_model = models[best_model_name]

          if best_model_score < 0.6:
              raise CustomException("No best model found")
          logging.info("Best model found")

          save_object(file_path = self.model_trainer_config.trained_model_file_path,
                      obj = best_model)
          
          predicted = best_model.predict(X_test)

          r2_ = r2_score(y_test, predicted)

          return r2_
        except Exception as e:
            raise CustomException(e, sys)
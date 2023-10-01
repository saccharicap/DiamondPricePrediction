import numpy as pd
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import sys
import os
from src.utils import evaluate_model


@dataclass

class ModelTrainerconfig:
   trained_model_file_path= os.path.join('artifacts','model.pkl')



class ModelTrainer:
   def __init__(self):
      self.model_trainer_config = ModelTrainerconfig()

   def initiate_model_training(self, train_array, test_array):
      try:

         logging.info('initiation of model training')

         logging.info('splittinf of dependent and independent feature from test_array, trainarray')

         X_train ,X_test, Y_train, Y_test =(train_array[:,:-1], test_array[:,:-1],train_array[:,-1],test_array[:,-1])

         logging.info("x_train x_test y_test Y_train data form")

         models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTreeRegressor':DecisionTreeRegressor()
     

         }
            
         model_report:dict=evaluate_model(X_train,X_test,Y_train,Y_test,models)
         print(model_report)
         print('\n=======================================================================')

         logging.info(f'model_report:{model_report}')

         ## to get best model score from dictionary
         best_model_score= max(sorted(model_report.values()))
         best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

         best_model= models[best_model_name]
         print(f'Best model found, model_name :{best_model_name}, r2 score:{best_model_score}')
         print('\n============================================================================')
         logging.info(f'best model found, model_name:{best_model_name}, r2score:{best_model_score}')


         save_object(file_path=self.model_trainer_config.trained_model_file_path, obj= best_model)


      except Exception as e:
         logging.info('initiate model training failed')
         raise CustomException(e,sys)
         

         



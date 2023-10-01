import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



def save_object(file_path, obj):
  try:

    dir_path= os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)


    with open( file_path,'wb') as file_obj:
        pickle.dump(obj,file_obj)
  except Exception as e:
    logging.info('pickle file not saved')
    raise CustomException(e,sys)



def evaluate_model(X_train,X_test,Y_train,Y_test, models):
   try:
      report={}


      for i in range(len(models)):
         model = list(models.values())[i]

         ## train_model
         model.fit(X_train,Y_train)

         ## model_prediction
         y_pred=model.predict(X_test)


         ## get r2_score  for train and test data
         ## train_model_score= r2_score(Y_train,y_pred)
         test_model_score= r2_score(Y_test, y_pred)

         report[list(models.keys())[i]] = test_model_score

      return report
      
   except Exception as e:
      logging('evaluation of model fail')

      raise CustomException(e,sys)
   

def load_object(file_path):
   try:
      
      with open(file_path,'rb') as file_obj:
         return pickle.load(file_obj)
      
   except Exception as e:
      logging.info('load_object failed')   
      raise CustomException(e,sys)
   

      

      




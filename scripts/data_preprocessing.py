from sklearn.model_selection import train_test_split
import pandas as pd
import os
from custom_logs import get_logger

logger = get_logger("preprocessing")

class preprocessingPipeline:
  def __init__(self,config):
    self.config = config
    os.makedirs(self.config.processed_data_folder, exist_ok=True)

  def load_data(self,raw_data_file):
    raw_data = pd.read_csv(raw_data_file)
    return raw_data

  def split_data(self,raw_data):
    X = raw_data.iloc[:,0:8]
    y = raw_data.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =self.config.test_size,random_state=self.config.random_state)

    return X_train,X_test,y_train,y_test

  def save_split_data(self, X_train, X_test, y_train, y_test, output_path):
      # Ensure the output path exists
      os.makedirs(output_path, exist_ok=True)

      # Combine features and labels for train and test data
      train_data = pd.concat([X_train, y_train], axis=1)
      test_data = pd.concat([X_test, y_test], axis=1)

      # Save to CSV
      train_csv_path = os.path.join(output_path, "train.csv")
      test_csv_path = os.path.join(output_path, "test.csv")

      train_data.to_csv(train_csv_path, index=False)
      test_data.to_csv(test_csv_path, index=False)

      logger.info(f"Train data saved to: {train_csv_path}")
      logger.info(f"Test data saved to: {test_csv_path}")
  
  def run_pipeline(self):

    raw_data=self.load_data(self.config.raw_data_file)
    logger.info("Raw data loaded")
    logger.info("Data Split Started")
    X_train,X_test,y_train,y_test = self.split_data(raw_data)
    logger.info("Split Done")
    logger.info("File Saving")
    self.save_split_data(X_train, X_test, y_train, y_test,self.config.processed_data_folder)
    
    
  
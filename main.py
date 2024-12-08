from scripts.data_preprocessing import  preprocessingPipeline
from scripts.config.data_preprocessing_config import DataPreprocessingConfig
from scripts.training import TrainingPipeline
from scripts.config.training_config import TrainingConfig

from custom_logs import get_logger
logger = get_logger("preprocessing.log")

#Data Preprocessing
logger.info("Data preprocessing Started.")
config=DataPreprocessingConfig()
pipe=preprocessingPipeline(config)
pipe.run_pipeline()
logger.info("Data preprocessing completed successfully.")

#Training
logger.info("Training Started")
config=TrainingConfig()
pipeline=TrainingPipeline(config)
pipeline.train_and_log()
logger.info("Training Done")
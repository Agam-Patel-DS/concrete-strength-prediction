from utils.common import read_yaml

params=read_yaml("configs/config.yaml")

class DataPreprocessingConfig:
    def __init__(self):
        """
        Initialize configuration for data preprocessing.

        Args:
        raw_data_file (str): Path to the raw data file.
        processed_data_folder (str): Path to the folder where processed data will be saved.
        """
        self.raw_data_file = params["data_preprocessing"]["raw_data_file"]
        self.processed_data_folder = params["data_preprocessing"]["processed_data_dir"]
        self.test_size =  params["data_preprocessing"]["test_size"]
        self.random_state =  params["data_preprocessing"]["random_state"]


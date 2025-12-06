from from_root import from_root
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from src.logger import LOG_DIR, LOG_FILE, logging




class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path
        

    def load_data(self):
        logging.info("loading data from csv file")
        try:
            df = pd.read_csv(self.data_path)
            logging.debug(f"Data read successfully from {self.data_path}")
            return df
        except Exception as e:
            logging.error(f"Error during data loading: {e}")
            raise

    def save_data(self, df: pd.DataFrame, file_path: str):
        logging.info(f"Saving data to {file_path}")
        try:
            df.to_csv(file_path, index=False)
            logging.debug(f"Data saved successfully to {file_path}")
        except Exception as e:
            logging.error(f"Error during saving data: {e}")
            raise


def main():
    data_ingestion = DataIngestion(data_path='C:\\Users\\Suraj\\Desktop\\coding\\mlproject\\bankruptcy prediction\\train.csv')
    
    # Load data
    df = data_ingestion.load_data()
    
    
    # Construct data path
    data_dir_path = os.path.join(from_root(), "Data")
    os.makedirs(data_dir_path, exist_ok=True)
    data_file_path = os.path.join(data_dir_path, "raw_data.csv")


    # Saving raw data
    data_ingestion.save_data(df, data_file_path)
    logging.info("raw_data saved successfully")

if __name__ == "__main__":
    main()
    
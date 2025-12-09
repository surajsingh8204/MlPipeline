import pandas as pd 
import os
from src.logger import LOG_DIR, LOG_FILE, logging

# Get project root directory (two levels up from this file: components -> src -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class DataCleaning:
    def __init__(self):
        pass

    def read_data(self, file_path: str) -> pd.DataFrame:
        logging.info(f"Reading data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            logging.debug(f"Data read successfully from {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error during reading data: {e}")
            raise

    def remove_unwanted_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logging.info("Removing unwanted columns from data")
        try:
            df = df.drop(columns=columns, errors="ignore")
            logging.debug(f"Unwanted columns {columns} removed successfully")
            return df
        except Exception as e:
            logging.error(f"Error during removing unwanted columns: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning process")
        try:
            # Example cleaning steps
            df = df.dropna()  # Remove missing values
            df = df.drop_duplicates()  # Remove duplicate rows
            logging.debug("Data cleaning completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
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
    data_cleaning = DataCleaning()
    
    # read data
    raw_data_path = os.path.join(PROJECT_ROOT, "Data", "raw_data.csv")
    df = data_cleaning.read_data(raw_data_path)

    # remove unwanted columns
    df = data_cleaning.remove_unwanted_columns(df, columns=["Unnamed: 0", "company_name"])  

    # clean data
    df = data_cleaning.clean_data(df) 

    #save cleaned data
    cleaned_data_path = os.path.join(PROJECT_ROOT, "Data", "cleaned_data.csv")
    data_cleaning.save_data(df, cleaned_data_path)  
    logging.info("cleaned data saved successfully")

if __name__ == "__main__":
    main()
    
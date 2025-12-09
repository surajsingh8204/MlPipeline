import pandas as pd 
import os
from src.logger import LOG_DIR, LOG_FILE, logging

# Get project root directory (two levels up from this file: components -> src -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataValidation:
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

    def validate_data(self, df: pd.DataFrame) -> bool:
        logging.info("Starting data validation process")
        try:
            # Example validation: Check for missing values
            if df.isnull().sum().sum() > 0:
                logging.warning("Data contains missing values")
                return False
            logging.debug("Data validation completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error during data validation: {e}")
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
    data_validation = DataValidation()
    
    # read data
    raw_data_path = os.path.join(PROJECT_ROOT, "Data", "raw_data.csv")
    df = data_validation.read_data(raw_data_path)

    # validate data
    is_valid = data_validation.validate_data(df)
    if not is_valid:
        logging.error("Data validation failed. Exiting process.")
        return

    # save validated data
    validated_data_path = os.path.join(PROJECT_ROOT, "Data", "validated_data.csv")
    data_validation.save_data(df, validated_data_path)
    logging.info("Data validation and saving completed successfully")

if __name__ == "__main__":
    main()
    
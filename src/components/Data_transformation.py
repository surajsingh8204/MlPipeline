import pandas as pd 
import os
import numpy as np

from pyparsing import col
from src.logger import LOG_DIR, LOG_FILE, logging

# Get project root directory (two levels up from this file: components -> src -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataTransformation:
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
    def map_label(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        logging.info(f"Applying label encoding to column: {column}")
        try:
            df[column] = df[column].map({"alive": 0, "failed": 1})
            logging.debug(f"Label encoding applied successfully to column: {column}")
            return df
        except Exception as e:
            logging.error(f"Error during label encoding: {e}")
            raise
    def one_hot_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to column: {column}")
        try:
            df = pd.get_dummies(df, columns=[column], drop_first=True)
            logging.debug(f"One-hot encoding applied successfully to column: {column}")
            return df
        except Exception as e:
            logging.error(f"Error during one-hot encoding: {e}")
            raise
    def frequency_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        logging.info(f"Applying frequency encoding to column: {column}")
        try:
            freq_encoding = df[column].value_counts(normalize=True)
            df[column] = df[column].map(freq_encoding)
            logging.debug(f"Frequency encoding applied successfully to column: {column}")
            return df
        except Exception as e:
            logging.error(f"Error during frequency encoding: {e}")
            raise
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting feature engineering process")
        try:
            eps = 1e-6  # to prevent division by zero

            df['Leverage_Ratio']      = df['X18'] / (df['X10'] + eps)
            df['Current_Ratio']       = df['X1'] / (df['X14'] + eps)
            df['Profit_Margin']       = df['X6'] / (df['X17'] + eps)
            df['Asset_Turnover']      = df['X17'] / (df['X10'] + eps)
            df['Debt_to_Equity']      = df['X18'] / (df['X10'] - df['X18'] + eps)
            df['EBIT_Margin']         = df['X11'] / (df['X17'] + eps)
            df['Gross_Margin']        = df['X13'] / (df['X17'] + eps)
            df['Receivables_Ratio']   = df['X7'] / (df['X10'] + eps)
            df['Inventory_Turnover']  = df['X2'] / (df['X5'] + eps)
            return df
        except Exception as e:
            logging.error(f"Error during feature engineering: {e}")
            raise
    
    def outliers_removal_winsorization(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Removing outliers from columns using winsorization")
        try:
            x_columns = [col for col in df.columns if col.startswith("X")]
            for column in x_columns:
                lower_bound = df[column].quantile(0.05)
                upper_bound = df[column].quantile(0.95)
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            logging.debug(f"Outliers removed successfully from column: {column}")
            return df
        except Exception as e:
            logging.error(f"Error during outlier removal: {e}")
            raise
    
    def scaling_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying standard scaling to numerical columns")
        try:
            from sklearn.preprocessing import StandardScaler
            ratio_features = [
                        'Leverage_Ratio', 'Current_Ratio', 'Profit_Margin', 'Asset_Turnover',
                        'Debt_to_Equity', 'EBIT_Margin', 'Gross_Margin',
                        'Receivables_Ratio', 'Inventory_Turnover'
                    ]
            scaler = StandardScaler()
            from sklearn.preprocessing import RobustScaler
            for col in ['Leverage_Ratio', 'Current_Ratio', 'Profit_Margin',
                'Asset_Turnover', 'EBIT_Margin', 'Gross_Margin',
                'Receivables_Ratio', 'Inventory_Turnover']:
                df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
                
                numeric_features = [col for col in df.columns if col.startswith("X")] + ratio_features

    
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
               
            return df
        except Exception as e:
            logging.error(f"Error during standard scaling: {e}")
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
    data_transformation = DataTransformation()
    
    # read data
    cleaned_data_path = os.path.join(PROJECT_ROOT, "Data", "cleaned_data.csv")

    df = data_transformation.read_data(cleaned_data_path)
    # map labeling
    df = data_transformation.map_label(df, column="status_label")  

    # one hot encoding
    df = data_transformation.one_hot_encoding(df, column="Division") 

    # frequency encoding
    df = data_transformation.frequency_encoding(df, column="MajorGroup")

   

    # feature engineering
    df = data_transformation.feature_engineering(df)

    # outliers removal using winsorization
    df = data_transformation.outliers_removal_winsorization(df)

    # standard scaling
    df = data_transformation.scaling_standardization(df)

    #save cleaned data
    transformed_data_path = os.path.join(PROJECT_ROOT, "Data", "transformed_data.csv")
    data_transformation.save_data(df, transformed_data_path)  
    logging.info("transformed data saved successfully")

if __name__ == "__main__":
    main()
    
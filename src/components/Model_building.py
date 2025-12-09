import pandas as pd 
import os
import numpy as np
from src.logger import LOG_DIR, LOG_FILE, logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Get project root directory (two levels up from this file: components -> src -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelBuilding:
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
    def split_features_target(self, df: pd.DataFrame, target_column: str) -> tuple:
        logging.info(f"Splitting features and target column: {target_column}")
        try:
            X = df.drop(columns=[target_column], errors="ignore")
            y = df[target_column]
            logging.debug("Features and target split successfully")
            return X, y
        except Exception as e:
            logging.error(f"Error during splitting features and target: {e}")
            raise

    def model_training(self, X: pd.DataFrame, y: pd.Series):
            logging.info("Starting model training process")
            try:
               
                model = XGBClassifier(n_estimators = 500,           
                                        max_depth = 6,               
                                        learning_rate = 0.05,
                                        subsample = 0.8,
                                        colsample_bytree = 0.8,
                                        random_state = 42,
                                        n_jobs = -1,
                                        use_label_encoder = False,
                                        eval_metric = "logloss")


                model.fit(X, y)
                logging.debug("Model trained successfully with XGBClassifier")
                return model
            except Exception as e:
                logging.error(f"Error during model training: {e}")
                raise 

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        logging.info("Evaluating model performance")
        try:
            accuracy = model.score(X, y)
            logging.debug(f"Model evaluation completed with accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise
    def save_model(self, model, file_path: str):
        import joblib
        logging.info(f"Saving model to {file_path}")
        try:
            joblib.dump(model, file_path)
            logging.debug(f"Model saved successfully to {file_path}")
        except Exception as e:
            logging.error(f"Error during saving model: {e}")
            raise

def main():
    model_building = ModelBuilding()
    
    # read data
    transformed_data_path = os.path.join(PROJECT_ROOT, "Data", "transformed_data.csv")

    df = model_building.read_data(transformed_data_path)
    # split features and target
    X, y = model_building.split_features_target(df, target_column="status_label")  

    # model training
    model = model_building.model_training(X, y) 

    # evaluate model
    accuracy = model_building.evaluate_model(model, X, y)
    logging.info(f"Random Forest Model Accuracy: {accuracy}")
   

    # Create Models directory if it doesn't exist
    models_dir = os.path.join(PROJECT_ROOT, "Models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    
    xgb_model_path = os.path.join(models_dir, "xgb_classifier_model.pkl")
    
    model_building.save_model(model, xgb_model_path)
    
    logging.info("Model building and saving completed successfully")

if __name__ == "__main__":
    main()         
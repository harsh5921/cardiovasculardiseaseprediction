import joblib
import pandas as pd
from preprocessing import preprocess_data

def predict(filepath, model_path):
    df = pd.read_csv(filepath)
    _, X_test, _, _ = preprocess_data(df)
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    return predictions

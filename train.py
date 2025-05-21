# train_all.py

import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from data_loading import load_data
from model import MODELS, PARAM_GRID

def train(algo="log_reg", output=None):
    try:
        # Load raw data
        df = load_data("data/cardio_train.csv")

        # Define features
        numeric_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

        # Define transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Separate features and label
        X = df.drop(columns=['cardio', 'id'])
        y = df['cardio']

        # Ensure X is still a DataFrame
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame"

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Load model and params
        model = MODELS[algo]
        param_grid = PARAM_GRID[algo]

        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        # Adjust param grid
        param_grid_pipeline = {f'classifier__{key}': value for key, value in param_grid.items()}

        # Grid search
        clf = GridSearchCV(pipeline, param_grid_pipeline, cv=3, scoring='accuracy', verbose=1)
        clf.fit(X_train, y_train)

        # Report
        print(f"\nBest params for {algo}: {clf.best_params_}")
        y_pred = clf.predict(X_test)
        print(f"\nClassification report for {algo}:\n{classification_report(y_test, y_pred, zero_division=0)}")

        # Save model
        if output:
            joblib.dump(clf.best_estimator_, output)
            print(f"✅ Model saved to {output}")
    except Exception as e:
        print("❌ Error occurred:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train different models on cardio dataset")
    parser.add_argument('--algo', type=str, choices=MODELS.keys(), default="log_reg", help="Algorithm to train")
    parser.add_argument('--output', type=str, help="Output path to save trained model")
    args = parser.parse_args()

    train(args.algo, args.output)

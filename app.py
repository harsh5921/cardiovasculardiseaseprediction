from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained HistGradientBoostingClassifier pipeline
model_path = 'models/hist_gb.joblib'  # Make sure this file exists
model = joblib.load(model_path)

# Feature names must match the ones used during training
FEATURE_NAMES = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Collect input values from form
            input_features = []
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)
                if value is None or value.strip() == '':
                    return render_template('index.html', error=f'Missing value for {feature}', features=FEATURE_NAMES)
                input_features.append(float(value))

            input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)

# Predict
            prediction = model.predict(input_df)
            result = 'At Risk' if prediction[0] == 1 else 'No Risk'

            return render_template('index.html', prediction=result, features=FEATURE_NAMES)

        except Exception as e:
            return render_template('index.html', error=str(e), features=FEATURE_NAMES)

    return render_template('index.html', features=FEATURE_NAMES)

if __name__ == '__main__':
    app.run(debug=True)

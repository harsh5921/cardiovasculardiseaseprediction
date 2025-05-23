from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'this_is_a_super_secret_key_5921'

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
            input_features = {}
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)
                if value is None or value == '':
                    return render_template('index.html', error=f'Missing value for {feature}', features=FEATURE_NAMES)
                input_features[feature] = float(value)

            input_array = pd.DataFrame([input_features], columns=FEATURE_NAMES)

            prediction = model.predict(input_array)
            result = 'At Risk' if prediction[0] == 1 else 'No Risk'

            session['prediction'] = result
            session['user_data'] = input_features
            return redirect(url_for('result'))

        except Exception as e:
            return render_template('index.html', error=str(e), features=FEATURE_NAMES)

    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/result')
def result():
    prediction = session.get('prediction')
    user_data = session.get('user_data')
    if not prediction or not user_data:
        return redirect(url_for('home'))
    return render_template('result.html', prediction=prediction, user_data=user_data)

if __name__ == '__main__':
    app.run(debug=True)

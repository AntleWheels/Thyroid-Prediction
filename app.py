from flask import Flask, request, jsonify, render_template
import numpy as np
from keras import models

app = Flask(__name__)
model = models.load_model('thyroid_detection_model.h5')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        def encode(val):
            return 1 if val.lower() in ['t', 'true', 'm', 'yes'] else 0

        input_features = [
        float(data['age']),
        encode(data['sex']),
        encode(data['on_thyroxine']),
        encode(data['query_on_thyroxine']),
        encode(data['on_antithyroid_medication']),
        encode(data['sick']),
        encode(data['pregnant']),
        float(data['TSH']),
        float(data['T3']),
        float(data['TT4']),
        float(data['T4U']),
        float(data['FTI']),
        encode(data['goitre']),
        encode(data['tumor']),
        encode(data['psych'])
        ]


        input_array = np.array([input_features]) 
        prediction = model.predict(input_array)[0]

        # Interpret model output (binary classification)
        predicted_class = "Positive" if prediction > 0.2 else "Negative"

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'student_model.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

FEATURES = ['leetcode_total', 'contest_rating', 'github_repos', 'github_commits', 'commit_streak', 'open_source_prs']

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        features = [float(data.get(f, 0)) for f in FEATURES]
        X = np.array([features])
        
        # Predict
        pred = model.predict(X)
        proba = model.predict_proba(X)[0]
        label = le.inverse_transform(pred)[0]
        
        # Build confidence scores per class
        classes = le.classes_.tolist()
        confidence = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, proba)}
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'top_confidence': round(float(max(proba)) * 100, 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import joblib
import os
import json
import webbrowser
import threading

app = Flask(__name__)
app.secret_key = "heart-disease-demo-secret"  # only used to pass result between pages

MODEL_PATH = 'heart_disease_model.pkl'

def train_and_save():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv('heart_disease_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    print(f'Model trained and saved to {MODEL_PATH}')
    return pipe

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print('Model loaded from disk ✓')
else:
    print('No saved model found — training now...')
    model = train_and_save()

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Human-readable labels used to echo back what the patient entered on the results page
CP_LABELS = {0: 'Typical angina', 1: 'Atypical angina', 2: 'Non-anginal pain', 3: 'No symptoms (asymptomatic)'}
RESTECG_LABELS = {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Probable / definite ventricular hypertrophy'}
SLOPE_LABELS = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
THAL_LABELS = {0: 'Unknown', 1: 'Normal', 2: 'Fixed defect', 3: 'Reversible defect'}
YESNO = {0: 'No', 1: 'Yes'}
SEX_LABELS = {0: 'Female', 1: 'Male'}

# Signature heartbeat line used on the hero of both pages
ECG_PATH = ("M 0,32 L 40,32 L 55,32 L 65,18 L 75,46 L 85,8 L 95,32 L 115,32 L 135,24 L 155,32 L 300,32 "
            "L 300,32 L 340,32 L 355,32 L 365,18 L 375,46 L 385,8 L 395,32 L 415,32 L 435,24 L 455,32 L 600,32 "
            "L 600,32 L 640,32 L 655,32 L 665,18 L 675,46 L 685,8 L 695,32 L 715,32 L 735,24 L 755,32 L 900,32 "
            "L 900,32 L 940,32 L 955,32 L 965,18 L 975,46 L 985,8 L 995,32 L 1015,32 L 1035,24 L 1055,32 L 1200,32")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', ecg_path=ECG_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    """Reads the form, runs the model, stores a friendly summary, and
    redirects the browser to a brand-new results page."""
    try:
        values = [float(request.form[f]) for f in FEATURE_NAMES]
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        prob_pct = round(probability * 100, 1)

        if probability >= 0.7:
            risk = 'High'
        elif probability >= 0.4:
            risk = 'Medium'
        else:
            risk = 'Low'

        raw = dict(zip(FEATURE_NAMES, [request.form[f] for f in FEATURE_NAMES]))

        summary_rows = [
            ('Age', f"{raw['age']} years"),
            ('Sex', SEX_LABELS[int(raw['sex'])]),
            ('Chest pain type', CP_LABELS[int(raw['cp'])]),
            ('Resting blood pressure', f"{raw['trestbps']} mm Hg"),
            ('Cholesterol', f"{raw['chol']} mg/dl"),
            ('Fasting blood sugar > 120 mg/dl', YESNO[int(raw['fbs'])]),
            ('Resting ECG result', RESTECG_LABELS[int(raw['restecg'])]),
            ('Max heart rate achieved', f"{raw['thalach']} bpm"),
            ('Exercise-induced chest pain', YESNO[int(raw['exang'])]),
            ('ST depression (oldpeak)', raw['oldpeak']),
            ('Slope of peak exercise ST segment', SLOPE_LABELS[int(raw['slope'])]),
            ('Major vessels colored by fluoroscopy', raw['ca']),
            ('Thalassemia result', THAL_LABELS[int(raw['thal'])]),
        ]

        result = {
            'prediction': prediction,
            'probability': prob_pct,
            'risk': risk,
            'label': 'Heart Disease Indicators Detected' if prediction == 1 else 'No Heart Disease Indicators Detected',
            'summary_rows': summary_rows,
        }
        session['result'] = json.dumps(result)
        return redirect(url_for('result'))

    except Exception as e:
        return render_template('index.html', error=str(e), ecg_path=ECG_PATH)


@app.route('/result')
def result():
    raw = session.get('result')
    if not raw:
        # Nobody submitted the form yet — send them back instead of a blank page
        return redirect(url_for('index'))
    result = json.loads(raw)
    return render_template('result.html', result=result, ecg_path=ECG_PATH)


@app.route('/model-info')
def model_info():
    """Return model metrics for the dashboard."""
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score
    from flask import jsonify

    df = pd.read_csv('heart_disease_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cm = confusion_matrix(y_test, y_pred).tolist()

    return jsonify({
        'test_accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'cv_mean': round(cv_scores.mean() * 100, 2),
        'cv_std': round(cv_scores.std() * 100, 2),
        'dataset_size': len(df),
        'feature_count': len(FEATURE_NAMES),
        'confusion_matrix': cm,
        'positive_cases': int(y.sum()),
        'negative_cases': int((y == 0).sum()),
    })


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f'\n  Opening {url} in your browser...\n')
    app.run(debug=False, port=5000)

from flask import Flask, render_template, request, send_from_directory,redirect,url_for
from preparing import DataFrameWrapper, FeatureSelector, OutlierCapper, BinaryMapper, SexMapper
import pandas as pd
import os
import pickle
import uuid
import warnings
import subprocess
import threading
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

# Predict using the loaded model


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'uploads'

# Load your model (make sure 'svm_pipeline.pkl' exists in the same directory)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

flag = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return render_template('prediction.html', error="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('prediction.html', error="No file selected.")

    try:
        # Try to read the file assuming it's UTF-8
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding='ISO-8859-1')
        except Exception as e:
            return f"Error reading file: {str(e)}", 400

    try:
        df['PhysicalHealthDays'] = df['PhysicalHealthDays'].fillna(0).astype(int)  # Fill NaN values with 0, then convert to int
        df['MentalHealthDays'] = df['MentalHealthDays'].fillna(0).astype(int)  # Same for PhysicalHealth
        y_true = df['HadHeartAttack']
        # Predict using the loaded model
        predictions = model.predict(df)
        df['Prediction'] = predictions
        y_pred = predictions
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"Accuracy: {accuracy}")
        # Map prediction values to Yes/No
        df['Prediction'] = df['Prediction'].map({1: 'Yes', 0: 'No'})
        # Optional: Rename columns for better readability
        column_rename_map = {
            'BMI': 'Body Mass Index',
            'Smoking': 'Smoker',
            'AlcoholDrinking': 'Drinks Alcohol',
            'Stroke': 'Has Stroke History',
            'PhysicalHealth': 'Days Physically Unwell',
            'MentalHealth': 'Days Mentally Unwell',
            'DiffWalking': 'Difficulty Walking',
            'Sex': 'Gender',
            'AgeCategory': 'Age Group',
            'Race': 'Ethnicity',
            'Diabetic': 'Diabetic',
            'PhysicalActivity': 'Physically Active',
            'GenHealth': 'General Health',
            'SleepTime': 'Sleep Hours',
            'Asthma': 'Asthma',
            'KidneyDisease': 'Kidney Disease',
            'SkinCancer': 'Skin Cancer',
            'Prediction': 'Heart Disease Prediction'
        }

        df.rename(columns=column_rename_map, inplace=True)

        # Optional: Human-friendly mapping for binary features
        binary_columns = [
            'Smoker', 'Drinks Alcohol', 'Has Stroke History', 'Difficulty Walking',
            'Physically Active', 'Asthma', 'Kidney Disease', 'Skin Cancer', 'HadHeartAttack',
            'PhysicalHealthDays', 'MentalHealthDays'
        ]
        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].map({1: 'Yes', 0: 'No', 'Yes': 'Yes', 'No': 'No'})

        # Save results
        result_filename = f"result_{uuid.uuid4().hex}.csv"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        df.to_csv(result_path, index=False)

        # Send preview to frontend
        data_preview = df.to_dict(orient='records')

        flag = True
        return render_template('prediction.html', results={
            'data': data_preview,
            'result_file': result_filename,
            'accuracy': accuracy
        })

    except Exception as e:
        return render_template('prediction.html', error=f"Error processing file: {str(e)}")

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    return redirect('http://localhost:8501')

def run_streamlit():
    subprocess.Popen(["streamlit", "run", "dashboard\dashboard.py","--server.port", "8501"])

# Start Streamlit in a separate thread
@app.before_request
def start_streamlit():
    threading.Thread(target=run_streamlit).start()

if __name__ == '__main__':
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
    app.run(debug=True, threaded=True, use_reloader=False) 



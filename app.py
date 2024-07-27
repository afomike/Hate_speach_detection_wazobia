from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the vectorizers
tfidf_vectorizers = {
    'english': pickle.load(open('vectorizers/english_tfidf.pkl', 'rb')),
    'yoruba': pickle.load(open('vectorizers/yoruba_tfidf.pkl', 'rb')),
    'hausa': pickle.load(open('vectorizers/hausa_tfidf.pkl', 'rb')),
    'igbo': pickle.load(open('vectorizers/igbo_tfidf.pkl', 'rb'))
}

# Load the models
models = {
    'english_hate': pickle.load(open('models/english_hate_Logistic Regression.pkl', 'rb')),
    'english_offensive': pickle.load(open('models/english_offensive_Logistic Regression.pkl', 'rb')),
    'yoruba_hate': pickle.load(open('models/yoruba_hate_Logistic Regression.pkl', 'rb')),
    'yoruba_offensive': pickle.load(open('models/yoruba_offensive_Logistic Regression.pkl', 'rb')),
    'hausa_hate': pickle.load(open('models/hausa_hate_Logistic Regression.pkl', 'rb')),
    'hausa_offensive': pickle.load(open('models/hausa_offensive_Logistic Regression.pkl', 'rb')),
    'igbo_hate': pickle.load(open('models/igbo_hate_Logistic Regression.pkl', 'rb')),
    'igbo_offensive': pickle.load(open('models/igbo_offensive_Logistic Regression.pkl', 'rb')),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    language = request.form['language']
    task = request.form['task']
    text = request.form['text']

    # Preprocess the text using the appropriate vectorizer
    vectorizer = tfidf_vectorizers[language]
    X_text = vectorizer.transform([text])

    # Load the appropriate model based on language and task
    model_key = f"{language}_{task}"
    model = models[model_key]

    # Make a prediction
    prediction = model.predict(X_text)[0]

    return f"Prediction: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load hate words dataset
hate_words_df = pd.read_csv('data/hate_words.csv')

# Load the trained models
with open('models/hate_speech_model.pkl', 'rb') as f:
    hate_model = pickle.load(f)

with open('models/offensive_speech_model.pkl', 'rb') as f:
    offensive_model = pickle.load(f)

def check_hate_speech(word):
    prediction = hate_model.predict([word])[0]
    if prediction == 1:
        reason = hate_words_df[hate_words_df['Hate word'] == word]['Why Hate?'].values
        return {
            'word': word,
            'is_hate_speech': True,
            'why_hate': reason[0] if len(reason) > 0 else "Reason not found in dataset"
        }
    return {
        'word': word,
        'is_hate_speech': False,
        'why_hate': None
    }

def check_offensive_speech(word):
    prediction = offensive_model.predict([word])[0]
    if prediction == 1:
        reason = hate_words_df[hate_words_df['Offensive'] == word]['Why offensive'].values
        return {
            'word': word,
            'is_offensive_speech': True,
            'why_offensive': reason[0] if len(reason) > 0 else "Reason not found in dataset"
        }
    return {
        'word': word,
        'is_offensive_speech': False,
        'why_offensive': None
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    word = request.form['word']
    hate_result = check_hate_speech(word)
    offensive_result = check_offensive_speech(word)
    result = {
        'word': word,
        'is_hate_speech': hate_result['is_hate_speech'],
        'why_hate': hate_result['why_hate'],
        'is_offensive_speech': offensive_result['is_offensive_speech'],
        'why_offensive': offensive_result['why_offensive']
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

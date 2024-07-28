from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


# Load the hate words datasets for each language
hate_words_datasets = {
    'english': pd.read_csv('data/english_hate_words.csv'),
    'yoruba': pd.read_csv('data/yoruba_hate_words.csv'),
    'hausa': pd.read_csv('data/hausa_hate_words.csv'),
    'igbo': pd.read_csv('data/igbo_hate_words.csv')
}

# Load the trained models for word detection (assuming models are language-specific as well)
hate_models = {
    'english': pickle.load(open('models/english_hate_speech_model.pkl', 'rb')),
    'yoruba': pickle.load(open('models/yoruba_hate_speech_model.pkl', 'rb')),
    'hausa': pickle.load(open('models/hausa_hate_speech_model.pkl', 'rb')),
    'igbo': pickle.load(open('models/igbo_hate_speech_model.pkl', 'rb'))
}

offensive_models = {
    'english': pickle.load(open('models/english_offensive_speech_model.pkl', 'rb')),
    'yoruba': pickle.load(open('models/yoruba_offensive_speech_model.pkl', 'rb')),
    'hausa': pickle.load(open('models/hausa_offensive_speech_model.pkl', 'rb')),
    'igbo': pickle.load(open('models/igbo_offensive_speech_model.pkl', 'rb'))
}

def check_hate_speech(word, language):
    model = hate_models[language]
    dataset = hate_words_datasets[language]
    prediction = model.predict([word])[0]
    if prediction == 1:
        reason = dataset[dataset['Hate word'] == word]['Why Hate?'].values
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

def check_offensive_speech(word, language):
    model = offensive_models[language]
    dataset = hate_words_datasets[language]
    prediction = model.predict([word])[0]
    if prediction == 1:
        reason = dataset[dataset['Offensive'] == word]['Why offensive'].values
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
    language = request.form['language']
    task = request.form['task']
    word = request.form['word']
    result = {}

    if task == 'hate':
        result = check_hate_speech(word, language)
    elif task == 'offensive':
        result = check_offensive_speech(word, language)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

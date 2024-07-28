from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the hate words datasets for each language
hate_words_datasets = {
    'yoruba': pd.read_csv('dataset/yoruba_hate.csv'),
    'hausa': pd.read_csv('dataset/hausa_hate.csv'),
    'igbo': pd.read_csv('dataset/igbo_hate.csv')
}

# Load the trained models for word detection (with updated paths)
hate_models = {
    'yoruba': pickle.load(open('models/yoruba_hate_speech_yoruba_KNN.pkl', 'rb')),
    'hausa': pickle.load(open('models/hausa_hate_speech_hausa_KNN.pkl', 'rb')),
    'igbo': pickle.load(open('models/igbo_hate_speech_igbo_KNN.pkl', 'rb'))
}

offensive_models = {
    'yoruba': pickle.load(open('models/yoruba_offensive_speech_yoruba_KNN.pkl', 'rb')),
    'hausa': pickle.load(open('models/hausa_offensive_speech_hausa_KNN.pkl', 'rb')),
    'igbo': pickle.load(open('models/igbo_offensive_speech_igbo_KNN.pkl', 'rb'))
}

# Load the vectorizers for each language (with updated paths)
tfidf_vectorizers = {
    'yoruba': pickle.load(open('models/yoruba_tfidf_vectorizer.pkl', 'rb')),
    'hausa': pickle.load(open('models/hausa_tfidf_vectorizer.pkl', 'rb')),
    'igbo': pickle.load(open('models/igbo_tfidf_vectorizer.pkl', 'rb'))
}

def check_hate_speech(word, language):
    model = hate_models[language]
    vectorizer = tfidf_vectorizers[language]
    X_word = vectorizer.transform([word])
    prediction = model.predict(X_word)[0]
    dataset = hate_words_datasets[language]
    if prediction == 1:
        reason = dataset[dataset['Hate words'] == word]['Why Hate?'].values
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
    vectorizer = tfidf_vectorizers[language]
    X_word = vectorizer.transform([word])
    prediction = model.predict(X_word)[0]
    dataset = hate_words_datasets[language]
    if prediction == 1:
        reason = dataset[dataset['Offensive words'] == word]['Why offensive?'].values
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
    print(task)
    if task == 'hate':
        result = check_hate_speech(word, language)
    elif task == 'offensive':
        result = check_offensive_speech(word, language)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

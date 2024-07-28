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

def check_speech(word, language):
    hate_model = hate_models[language]
    offensive_model = offensive_models[language]
    vectorizer = tfidf_vectorizers[language]
    X_word = vectorizer.transform([word])
    
    hate_prediction = hate_model.predict(X_word)[0]
    offensive_prediction = offensive_model.predict(X_word)[0]
    dataset = hate_words_datasets[language]
    
    result = {'word': word, 'is_hate_speech': False, 'why_hate': None, 'is_offensive_speech': False, 'why_offensive': None}
    
    if hate_prediction == 1:
        reason_hate = dataset[dataset['Hate words'] == word]['Why Hate?'].values
        result['is_hate_speech'] = True
        result['why_hate'] = reason_hate[0] if len(reason_hate) > 0 else "Reason not found in dataset"
        
    if offensive_prediction == 1:
        reason_offensive = dataset[dataset['Offensive words'] == word]['Why offensive?'].values
        result['is_offensive_speech'] = True
        result['why_offensive'] = reason_offensive[0] if len(reason_offensive) > 0 else "Reason not found in dataset"
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    language = request.form['language']
    word = request.form['word']
    result = check_speech(word, language)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

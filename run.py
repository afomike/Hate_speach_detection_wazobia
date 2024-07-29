from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def find_similar_word(word, words_list, vectorizer):
    # Vectorize the words
    word_vec = vectorizer.transform([word])
    words_vecs = vectorizer.transform(words_list)
    
    # Compute cosine similarity
    similarities = cosine_similarity(word_vec, words_vecs).flatten()
    
    # Find the index of the most similar word
    most_similar_index = np.argmax(similarities)
    
    return words_list[most_similar_index], similarities[most_similar_index]

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
        if len(reason_hate) > 0:
            result['is_hate_speech'] = True
            result['why_hate'] = reason_hate[0]
        else:
            similar_word, similarity = find_similar_word(word, dataset['Hate words'].tolist(), vectorizer)
            reason_hate = dataset[dataset['Hate words'] == similar_word]['Why Hate?'].values
            result['is_hate_speech'] = True
            result['why_hate'] = f"Similar to '{similar_word}' (similarity: {similarity:.2f}): {reason_hate[0]}"
        
    if offensive_prediction == 1:
        reason_offensive = dataset[dataset['Offensive words'] == word]['Why offensive?'].values
        if len(reason_offensive) > 0:
            result['is_offensive_speech'] = True
            result['why_offensive'] = reason_offensive[0]
        else:
            similar_word, similarity = find_similar_word(word, dataset['Offensive words'].tolist(), vectorizer)
            reason_offensive = dataset[dataset['Offensive words'] == similar_word]['Why offensive?'].values
            result['is_offensive_speech'] = True
            result['why_offensive'] = f"Similar to '{similar_word}' (similarity: {similarity:.2f}): {reason_offensive[0]}"
    
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

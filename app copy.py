from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load models
models = {
    'yoruba': {
        'hate': 'models/yoruba_hate_speech_yoruba_KNN.pkl',
        'offensive': 'models/yoruba_offensive_speech_yoruba_KNN.pkl',
        'tfidf': 'models/yoruba_tfidf_vectorizer.pkl'
    },
    'hausa': {
        'hate': 'models/hausa_hate_speech_hausa_KNN.pkl',
        'offensive': 'models/hausa_offensive_speech_hausa_KNN.pkl',
        'tfidf': 'models/hausa_tfidf_vectorizer.pkl'
    },
    'igbo': {
        'hate': 'models/igbo_hate_speech_igbo_KNN.pkl',
        'offensive': 'models/igbo_offensive_speech_igbo_KNN.pkl',
        'tfidf': 'models/igbo_tfidf_vectorizer.pkl'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    language = request.form['language']
    task = request.form['task']
    word = request.form['word']
    
    # Load the appropriate model and vectorizer
    model_path = models[language][task]
    tfidf_path = models[language]['tfidf']
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(tfidf_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Transform the word
    word_tfidf = tfidf_vectorizer.transform([word])
    
    # Predict
    prediction = model.predict(word_tfidf)
    
    response = {'word': word}
    
    if task == 'hate':
        response['is_hate_speech'] = bool(prediction[0])
        # Assuming 'Why Hate?' is a column in the original data
        response['why_hate'] = "Assumed reason"  # Placeholder, replace with actual logic if needed
    elif task == 'offensive':
        response['is_offensive_speech'] = bool(prediction[0])
        # Assuming 'Why Offensive?' is a column in the original data
        response['why_offensive'] = "Assumed reason"  # Placeholder, replace with actual logic if needed
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

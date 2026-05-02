# 🛡️ NaijaDetect — Nigerian Language Hate & Offensive Speech Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-Yoruba%20%7C%20Hausa%20%7C%20Igbo-purple?style=flat-square)

> The first machine learning tool that classifies hate and offensive speech across **Yoruba**, **Hausa**, and **Igbo** — Nigeria's three most widely spoken languages.

🚀 **[Live Demo → hate-speach-detection-wazobia.onrender.com](https://hate-speach-detection-wazobia.onrender.com/)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Models](#training-the-models)
  - [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Dataset Structure](#dataset-structure)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**NaijaDetect** is a web-based NLP application that detects hate speech and offensive speech in Nigerian languages. It uses language-specific K-Nearest Neighbours (KNN) classifiers trained on curated word-level datasets for Yoruba, Hausa, and Igbo. The app is served via a Flask backend and features a clean, real-time frontend interface.

This project addresses a significant gap in African language NLP research, where content moderation tools have historically been limited to English and other widely-resourced languages.

---

## Features

- 🌍 **3 Nigerian Languages** — Dedicated trained models for Yoruba, Hausa, and Igbo
- 🎯 **Dual Classification Tasks** — Separate detection modes for hate speech and offensive speech
- ⚡ **Real-Time Detection** — Instant classification results via AJAX (no page reload)
- 🧠 **Explainable Results** — Every positive detection returns a human-readable explanation
- 🔒 **Extensible Architecture** — Easily plug in new languages or swap classifiers

---

## Project Structure

```
NaijaDetect/
│
├── app.py                          # Flask application — routes and model inference
│
├── templates/
│   ├── index.html                  # Main UI — detection form and results
│   └── result.html                 # Standalone result page (legacy)
│
├── static/
│   └── styles.css                  # Application stylesheet
│
├── dataset/
│   ├── yoruba_hate.csv             # Yoruba labelled hate/offensive word dataset
│   ├── hausa_hate.csv              # Hausa labelled hate/offensive word dataset
│   └── igbo_hate.csv               # Igbo labelled hate/offensive word dataset
│
├── models/
│   ├── yoruba_hate_speech_yoruba_KNN.pkl
│   ├── yoruba_offensive_speech_yoruba_KNN.pkl
│   ├── hausa_hate_speech_hausa_KNN.pkl
│   ├── hausa_offensive_speech_hausa_KNN.pkl
│   ├── igbo_hate_speech_igbo_KNN.pkl
│   ├── igbo_offensive_speech_igbo_KNN.pkl
│   ├── yoruba_tfidf_vectorizer.pkl
│   ├── hausa_tfidf_vectorizer.pkl
│   └── igbo_tfidf_vectorizer.pkl
│
├── train_yoruba.ipynb              # Model training notebook — Yoruba
├── train_hausa.ipynb               # Model training notebook — Hausa
├── train_igbo.ipynb                # Model training notebook — Igbo
│
├── README.md
└── LICENSE
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.8+, Flask |
| Machine Learning | scikit-learn (KNN, TF-IDF), pickle |
| NLP Preprocessing | NLTK, WordNetLemmatizer |
| Frontend | HTML5, CSS3, Vanilla JavaScript (Fetch API) |
| Notebooks | Jupyter Notebook |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) A virtual environment tool such as `venv` or `conda`

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/hate-speach-detection-wazobia.git
   cd hate-speach-detection-wazobia
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install flask pandas scikit-learn nltk wordcloud jupyter
   ```

4. **Download NLTK data** (first-time setup)

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

---

### Training the Models

If you do not have pre-trained models, run the training notebooks in order. Each notebook reads from the corresponding CSV dataset, preprocesses the text, fits a TF-IDF vectorizer and a KNN classifier, and serialises all artefacts to the `models/` directory.

```bash
jupyter notebook train_yoruba.ipynb
jupyter notebook train_hausa.ipynb
jupyter notebook train_igbo.ipynb
```

After training, the `models/` directory should contain nine `.pkl` files — three KNN models for hate speech, three for offensive speech, and three TF-IDF vectorisers (one per language).

---

### Running the App

```bash
python app.py
```

The Flask development server starts on `http://127.0.0.1:5000` by default. Open this URL in your browser to use the detection interface.

For a production deployment, use a WSGI server such as Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 app:app
```

---

## How It Works

```
User Input (word/phrase)
        │
        ▼
  Language Selection ──► Load language-specific TF-IDF Vectorizer
        │
        ▼
  Task Selection ──────► Load KNN model (hate / offensive)
        │
        ▼
  TF-IDF Transform ──► vectorizer.transform([word])
        │
        ▼
  KNN Prediction ──────► model.predict(X_word)
        │
        ▼
  If prediction == 1:
    Lookup explanation in CSV dataset
        │
        ▼
  JSON response ──────► Frontend renders result with badge + explanation
```

1. The user selects a **language** and a **task** (hate or offensive detection).
2. The input word is transformed using the pre-fitted **TF-IDF vectoriser** for that language.
3. The transformed vector is passed to the **KNN classifier**, which returns a binary prediction (`1` = flagged, `0` = safe).
4. If flagged, the app looks up the corresponding row in the dataset CSV to retrieve a human-readable explanation.
5. The result is returned as JSON and rendered in the UI without a page reload.

---

## API Reference

### `POST /detect`

Classifies a word or phrase for hate or offensive speech.

**Request — `multipart/form-data`**

| Field | Type | Values | Description |
|---|---|---|---|
| `language` | string | `yoruba`, `hausa`, `igbo` | Language of the input word |
| `task` | string | `hate`, `offensive` | Classification task |
| `word` | string | any | The word or phrase to analyse |

**Response — Hate Speech Task**

```json
{
  "word": "example",
  "is_hate_speech": true,
  "why_hate": "This word targets a specific ethnic group."
}
```

**Response — Offensive Speech Task**

```json
{
  "word": "example",
  "is_offensive_speech": false,
  "why_offensive": null
}
```

---

## Dataset Structure

Each CSV dataset (`yoruba_hate.csv`, `hausa_hate.csv`, `igbo_hate.csv`) follows this schema:

| Column | Description |
|---|---|
| `Hate words` | The flagged hate word or phrase |
| `Why Hate?` | Human-annotated explanation for the hate classification |
| `Offensive words` | The flagged offensive word or phrase |
| `Why offensive?` | Human-annotated explanation for the offensive classification |

Missing values are imputed with the column mode during the training phase.

---

## Model Details

| Language | Algorithm | Vectoriser | Tasks |
|---|---|---|---|
| Yoruba | K-Nearest Neighbours | TF-IDF | Hate Speech, Offensive Speech |
| Hausa | K-Nearest Neighbours | TF-IDF | Hate Speech, Offensive Speech |
| Igbo | K-Nearest Neighbours | TF-IDF | Hate Speech, Offensive Speech |

**Preprocessing pipeline (per language):**

- Lowercasing
- Punctuation removal (configurable)
- Stopword removal and lemmatisation (configurable — see notebooks)
- TF-IDF vectorisation with per-language vocabulary

Additional classifiers evaluated during training include Logistic Regression, Decision Tree, Random Forest, SVM, and Multinomial Naive Bayes. The KNN model was selected for deployment.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure any new language models include a corresponding training notebook and dataset CSV following the schema described above.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

*Built with Python, Flask & Machine Learning · Nigerian Language NLP Research · © 2025 NaijaDetect*

from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
import pdfplumber
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Téléchargement des ressources nécessaires pour NLTK
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Chargement du modèle et du tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)
    sentiment = "positif" if probabilities[0][1] > probabilities[0][0] else "négatif"
    confidence = max(probabilities[0]).item()
    return sentiment, confidence

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Gestion des textes insérés par l'utilisateur
        user_text = request.form.get('user_text', '')
        if user_text:
            sentiment, confidence = analyze_sentiment(user_text)
            flash(f'Sentiment: {sentiment} (Confiance: {confidence:.2f})')

        # Gestion des fichiers PDF téléversés
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.pdf'):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                pdf_text = extract_text_from_pdf(filename)
                sentiment, confidence = analyze_sentiment(pdf_text)
                flash(f'Analyse PDF: Sentiment: {sentiment} (Confiance: {confidence:.2f})')
            else:
                flash('Veuillez télécharger un fichier PDF valide.')
        
        return redirect(url_for('index'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


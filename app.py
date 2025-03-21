from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from sentiment_analysis import (
    extract_text_from_pdf, 
    analyze_document, 
    summarize_text, 
    extract_entities, 
    detect_tasks, 
    suggest_response, 
    evaluate_complexity,
    classify_intentions  # Fonctionnalité potentielle ajoutée
)
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = None
    summary = None
    entities = None
    tasks = None
    response_suggestions = None
    complexity_evaluation = None
    intentions = None  # Pour la classification des intentions

    if request.method == 'POST':
        if 'file' in request.files and request.files['file']:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                text = extract_text_from_pdf(filepath)
                analysis = analyze_document(text)
                summary = summarize_text(text)
                entities = extract_entities(text)
                tasks = detect_tasks(text)
                response_suggestions = suggest_response(analysis)
                complexity_evaluation = evaluate_complexity(text)
                intentions = classify_intentions(text)  # Utilisation de la fonctionnalité d'intention

    return render_template('index.html', analysis=analysis, summary=summary, entities=entities, tasks=tasks, response_suggestions=response_suggestions, complexity_evaluation=complexity_evaluation, intentions=intentions)

if __name__ == '__main__':
    app.run(debug=True)


from transformers import pipeline, CamembertTokenizer, CamembertForTokenClassification

# Initialisation des pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Utilisation explicite de CamemBERT pour le NER sans conversion tiktoken
ner_model = CamembertForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
ner_tokenizer = CamembertTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

def extract_text_from_pdf(pdf_path):
    import pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = [page.extract_text() for page in pdf.pages[:3] if page.extract_text()]
        return "\n\n".join(full_text)
    except Exception as e:
        return "Erreur: Le fichier n'est pas un PDF valide."

def analyze_document(text):
    if not text:
        return "Aucun contenu à analyser."
    outputs = sentiment_pipeline(text, truncation=True)
    positive = sum(1 for output in outputs if 'positive' in output['label'].lower())
    negative = sum(1 for output in outputs if 'negative' in output['label'].lower())
    return interpret_results(positive, negative, len(outputs))

def interpret_results(positive, negative, total):
    if positive > negative:
        return "L'analyse révèle un engagement positif potentiel du client."
    elif negative > positive:
        return "L'analyse suggère que le client pourrait ne pas honorer ses engagements."
    else:
        return "L'analyse montre un mélange équilibré de sentiments. Suivi conseillé."

def summarize_text(text):
    max_len = 130
    summary = summarization_pipeline(text, max_length=max_len, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_entities(text):
    entities = ner_pipeline(text)
    unique_entities = {ent['word']: ent['entity_group'] for ent in entities}
    return list(unique_entities.items())

def detect_tasks(text):
    return summarize_text(text)

def suggest_response(analysis):
    if "négatif" in analysis:
        return "Considérez de rassurer le client avec des offres ou garanties."
    elif "positif" in analysis:
        return "Envoyez des remerciements et proposez plus de services."
    else:
        return "Réévaluez la situation avant de répondre."

def evaluate_complexity(text):
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    word_count = len(text.split())
    sentence_length = word_count / max(num_sentences, 1)
    return f"Longueur moyenne des phrases: {sentence_length:.2f} mots."

def classify_intentions(text):
    candidate_labels = ["achat", "plainte", "demande", "remboursement", "information"]
    result = intention_pipeline(text, candidate_labels=candidate_labels, multi_class=True)
    return {label: score for label, score in zip(result['labels'], result['scores'])}


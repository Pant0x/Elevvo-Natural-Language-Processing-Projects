from flask import Flask, render_template, request
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os

# --- Flask app setup ---
app = Flask(__name__)

# --- Paths to model files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models")

# --- Load models and vectorizer ---
logistic_path = os.path.join(MODEL_PATH, "model_logistic.pkl")
nb_path = os.path.join(MODEL_PATH, "model_nb.pkl")
vectorizer_path = os.path.join(MODEL_PATH, "vectorizer.pkl")

model_logistic = pickle.load(open(logistic_path, 'rb'))
model_nb = pickle.load(open(nb_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# --- NLTK setup ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# --- Text preprocessing ---
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in stop_words]
    return " ".join(filtered)

# --- Sentiment prediction helper ---
def get_sentiment(text: str, model_choice: str = "logistic") -> str:
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    model_choice = model_choice.lower()
    if model_choice == "logistic":
        return model_logistic.predict(vect)[0]
    elif model_choice == "naive_bayes":
        return model_nb.predict(vect)[0]
    else:
        return "Invalid model selection."

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment, review_text, selected_model = None, "", "logistic"

    if request.method == "POST":
        review_text = request.form.get("review", "")
        selected_model = request.form.get("model_choice", "logistic")
        sentiment = get_sentiment(review_text, selected_model)

    return render_template(
        "index.html",
        sentiment=sentiment,
        review=review_text,
        model_choice=selected_model,
    )

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)

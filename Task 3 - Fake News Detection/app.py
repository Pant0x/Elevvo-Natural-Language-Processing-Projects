from flask import Flask, render_template, request
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

app = Flask(__name__)

# Load models and vectorizer
model_logistic = pickle.load(open('model_logistic.pkl', 'rb'))
model_nb = pickle.load(open('model_nb.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    review = ''
    model_choice = 'logistic'
    if request.method == 'POST':
        review = request.form['review']
        model_choice = request.form.get('model_choice', 'logistic')
        processed = preprocess_text(review)
        vect = vectorizer.transform([processed])
        if model_choice == 'logistic':
            sentiment = model_logistic.predict(vect)[0]
        elif model_choice == 'naive_bayes':
            sentiment = model_nb.predict(vect)[0]
        else:
            sentiment = 'Invalid model selection.'
    return render_template('index.html', sentiment=sentiment, review=review, model_choice=model_choice)

if __name__ == '__main__':
    app.run(debug=True)

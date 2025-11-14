import pickle
from flask import Flask, request, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load models
models = {
    'logistic': pickle.load(open('model/logistic_regression_model.pkl', 'rb')),
    'naive_bayes': pickle.load(open('model/naive_bayes_model.pkl', 'rb')),
    'random_forest': pickle.load(open('model/random_forest_model.pkl', 'rb')),
    'svm': pickle.load(open('model/svm_model.pkl', 'rb'))
}

# Load vectorizer
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb')) if 'tfidf_vectorizer.pkl' in models else None

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
    category = None
    model_choice = 'logistic'
    news_text = ''
    if request.method == 'POST':
        news_text = request.form['news_text']
        model_choice = request.form['model_choice']
        processed_text = preprocess_text(news_text)
        vect_text = vectorizer.transform([processed_text])
        pred = models[model_choice].predict(vect_text)[0]
        category = pred
    return render_template('index.html', category=category, model_choice=model_choice, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)

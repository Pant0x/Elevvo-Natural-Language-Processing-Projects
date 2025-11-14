# Elevvo-NLP-Projects-Portfolio

This repository contains a series of Natural Language Processing (NLP) projects covering sentiment analysis, text classification, fake news detection, named entity recognition, topic modeling, question answering, text summarization, and document similarity. Each project is designed to build practical skills in text preprocessing, feature engineering, model training, evaluation, and visualization using real-world datasets.

---

## Table of Contents

1. [Sentiment Analysis on Product Reviews](#task-1-sentiment-analysis-on-product-reviews)  
2. [News Category Classification](#task-2-news-category-classification)  
3. [Fake News Detection](#task-3-fake-news-detection)  
4. [Named Entity Recognition (NER) from News Articles](#task-4-named-entity-recognition-ner-from-news-articles)  
5. [Topic Modeling on News Articles](#task-5-topic-modeling-on-news-articles)  
6. [Question Answering with Transformers](#task-6-question-answering-with-transformers)  
7. [Text Summarization Using Pre-trained Models](#task-7-text-summarization-using-pre-trained-models)  
8. [Resume Screening Using NLP](#task-8-resume-screening-using-nlp)  

---

## Task 1: Sentiment Analysis on Product Reviews

**Description:** Analyze product reviews to determine whether the sentiment is positive or negative.

**Steps:**
- Clean and preprocess text (lowercasing, removing stopwords)  
- Convert text to numerical format using TF-IDF or CountVectorizer  
- Train a binary classifier (e.g., Logistic Regression)  
- Evaluate performance using accuracy, precision, recall, and F1-score  

**Tools & Libraries:** Python, Pandas, NLTK, spaCy, Scikit-learn  

**Bonus:** Completed extra tasks to explore additional techniques:  
- Visualize most frequent positive and negative words  
- Train using a simple neural network (Keras)  
- Compare accuracy with Naive Bayes classifier  

---

## Task 2: News Category Classification

**Description:** Classify news articles into categories such as sports, business, politics, and technology.

**Steps:**
- Preprocess text (tokenization, stopword removal, lemmatization)  
- Vectorize text using TF-IDF or word embeddings  
- Train a multiclass classifier (Logistic Regression, Random Forest, or SVM)  

**Tools & Libraries:** Python, Pandas, Scikit-learn  

**Bonus:** Completed optional tasks to dig deeper:  
- Experiment with XGBoost or LightGBM  
- Visualize most frequent words per category using bar plots or word clouds  

---

## Task 3: Fake News Detection

**Description:** Classify news articles as real or fake based on text content.

**Steps:**
- Preprocess title and content (remove stopwords, lemmatize, vectorize)  
- Train a logistic regression or SVM classifier  
- Evaluate using accuracy and F1-score  

**Tools & Libraries:** Python, spaCy, Pandas, Scikit-learn  

**Bonus:** Completed extra tasks for additional insight:  
- Visualize extracted entities with displaCy  
- Compare results using different spaCy models  

---

## Task 4: Named Entity Recognition (NER) from News Articles

**Description:** Identify named entities (people, locations, organizations) from news articles.

**Steps:**
- Apply rule-based and model-based NER approaches  
- Highlight and categorize extracted entities in the text  

**Tools & Libraries:** Python, spaCy, Pandas  

**Bonus:** Completed optional tasks to extend learning:  
- Compare NER performance using different spaCy models  
- Visualize entities in text with colors and labels  

---

## Task 5: Topic Modeling on News Articles

**Description:** Discover hidden topics or themes in a collection of news articles.

**Steps:**
- Preprocess text (tokenization, lowercasing, stopword removal)  
- Apply Latent Dirichlet Allocation (LDA)  
- Display the most significant words per topic  

**Tools & Libraries:** Python, Gensim, pyLDAvis, NLTK, spaCy, Scikit-learn  

**Bonus:** Completed extra tasks to explore alternative approaches:  
- Compare LDA vs. NMF performance  
- Visualize topic-word distributions with pyLDAvis or word clouds  

---

## Task 6: Question Answering with Transformers

**Description:** Build a system that answers questions based on a given passage.

**Steps:**
- Use pre-trained transformer models (BERT, DistilBERT) fine-tuned for question answering  
- Feed the model both context and question to extract the answer span  
- Evaluate using exact match and F1 score  

**Tools & Libraries:** Hugging Face Transformers, Python, ROUGE  

**Bonus:** Completed optional tasks to go further:  
- Try extractive summarization using TextRank or Gensim  
- Fine-tune a pre-trained summarizer on custom datasets  

---

## Task 7: Text Summarization Using Pre-trained Models

**Description:** Generate concise summaries from long documents.

**Steps:**
- Preprocess and truncate texts to model input limits  
- Use encoder-decoder architectures (T5, BART, Pegasus)  
- Evaluate summaries with ROUGE scores  

**Tools & Libraries:** Python, Hugging Face Transformers  

**Bonus:** Completed extra tasks for deeper experimentation:  
- Compare abstractive vs extractive summarization  
- Experiment with fine-tuning on custom datasets  

---

## Task 8: Resume Screening Using NLP

**Description:** Screen and rank resumes based on job descriptions.

**Steps:**
- Preprocess resumes and job descriptions using embeddings  
- Compute similarity scores (cosine similarity or classification)  
- Present top-ranked resumes with match scores  

**Tools & Libraries:** Python, Sentence Transformers, Pandas, Scikit-learn  

**Bonus:** Completed extra tasks for better insight:  
- Build a simple front-end or Streamlit app for resume upload and results  
- Extract named entities (skills, experience) from resumes  

---

## Notes
- Each project includes full documentation of data exploration, preprocessing, model training, and evaluation.  
- Bonus tasks are optional, but I completed them where it added extra learning value.  
- Datasets are publicly available on Kaggle or other open sources.

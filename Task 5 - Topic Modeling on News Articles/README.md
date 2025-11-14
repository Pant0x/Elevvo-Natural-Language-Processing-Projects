## 🔍 News Topic Modelling App
This project is a **Natural Language Processing (NLP)** application that performs **topic modelling** on news articles using **LDA (Latent Dirichlet Allocation)** and **NMF (Non-negative Matrix Factorization)**. It helps automatically discover hidden themes and categorize news content.

---

## 🧩 Key Steps
- **Data Handling & Preprocessing**  
  - Tokenization, stopword removal, lemmatization, et.c 
  - Cleaning text using `BeautifulSoup` and `re`  
- **Feature Extraction**  
  - Using **CountVectorizer** or **TF-IDF Vectorizer**  
- **Model Training & Topic Modelling**  
  - Latent Dirichlet Allocation (LDA)  
  - Non-negative Matrix Factorization (NMF)  
  - Compare performance between LDA and NMF  
- **Visualization**  
  - Top words per topic (WordClouds)  
  - Interactive topic-word visualization using **pyLDAvis**  
- **Deployment**  
  - Built a **Streamlit app** for interactive prediction  
  - Users can select LDA or NMF and input news articles to get predicted topics

---

## 📂 Dataset
The dataset used is the BBC News Dataset (40,000+ samples).
- Available on:
  - [Kaggle - BBC News Dataset](https://www.kaggle.com/datasets/gpreda/bbc-news)
  - [Google Drive - BBC News Dataset](https://drive.google.com/file/d/14OXcLK6HNOJOa6iJtR8bzVG9kwcHLVzt/view?usp=drive_link)

---

## 📊 Visualizations
- WordClouds for the most significant words per topic  
- Interactive pyLDAvis visualizations for topic-word distributions  
- Compare LDA vs NMF for dominant topic extraction   

---

## 🧠 Tech Stack & Tools
- **Python Libraries**:  
  `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `WordCloud`, `BeautifulSoup`, `NLTK`, `Scikit-learn`, `pyLDAvis`  
- **Deployment**: Streamlit for interactive prediction  
- **Others**: GitHub / Google Colab / Kaggle for experimentation

---

## 📦 Dependencies
Before running locally, ensure these are installed:

```sh
pip install streamlit numpy pandas matplotlib seaborn wordcloud beautifulsoup4 nltk scikit-learn pyLDAvis joblib
```

## Installing
To install Streamlit:
```sh
pip install streamlit
```
To install all required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Application Locally
```sh
streamlit run app.py
```
Then open the local URL (usually http://localhost:8501/) in your browser.

## 📰 Try the App Online
You can use the app directly here: [News Articles Modelling](https://news-articles-modelling.streamlit.app/)<br>
Simply paste or type any news article or headline in the input box and click Analyze to discover its dominant topic (e.g., Sports, Conflicts, Politics, Health, etc.).

---

## 💡 Features
- Clean and preprocess news text data
- Extract meaningful features using CountVectorizer or TF-IDF
- Train and compare LDA and NMF topic models
- Visualize top words and topic-word distributions
- Deploy an interactive Streamlit app for prediction

---

## 📂 Folder Structure
```
Topic-Modelling-News-Articles/
├── app.py               
├── lda_pipeline.joblib  
├── nmf_pipeline.joblib  
├── requirements.txt     
├── images/              
│   ├── lda_word_cloud
│   ├──...
│   ├── nmf_word_cloud
│   ├──...
└── README.md
```

---

## ❓ Help
If you encounter any issues:
- Check the [Streamlit Documentation](https://docs.streamlit.io/)
- Search for similar issues or solutions on [Kaggle](https://www.kaggle.com/)
- Open an issue in this repository

---

## ✍️ Author
👤 Oluyale Ezekiel
- 📧 Email: ezekieloluyale@gmail.com
- LinkedIn: [Ezekiel Oluyale](https://www.linkedin.com/in/ezekiel-oluyale)
- GitHub Profile: [@amusEcode1](https://github.com/amusEcode1)
- Twitter: [@amusEcode1](https://x.com/amusEcode1?t=uHxhLzrA1TShRiSMrYZQiQ&s=09)

---

## 🙏 Acknowledgement
Thank you, Elevvo, for the incredible opportunity and amazing Internship.

import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load models
@st.cache_resource
def load_models():
    lda_pipeline = joblib.load("lda_pipeline.joblib")
    nmf_pipeline = joblib.load("nmf_pipeline.joblib")
    return lda_pipeline, nmf_pipeline

lda_pipeline, nmf_pipeline = load_models()

# App Header
st.title("🔍 News Topic Modelling App")
st.write("Classify news articles into topics using LDA or NMF")

# Sidebar: Model selection
model_choice = st.sidebar.selectbox("Choose Topic Modelling Algorithm", ["LDA", "NMF"])

# User input
user_text = st.text_area("Enter a News Article:")

# Predict Topic
if st.button("Predict Topic"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Select pipeline
        pipeline = lda_pipeline if model_choice == "LDA" else nmf_pipeline
        
        # Transform input text
        topic_distribution = pipeline.transform([user_text])
        topic_index = topic_distribution.argmax()
      
        # Get topic label from pipeline
        topic_labels = getattr(pipeline, "topic_labels", None)
        if topic_labels:
            label_name = topic_labels.get(topic_index, f"Topic {topic_index + 1}")
        else:
            label_name = f"Topic {topic_index + 1}"
        
        # Display predicted topic
        st.success(f"Predicted Topic: {label_name}")
        st.write("Topic probabilities:", topic_distribution)
    
        # Display Top Words for Topic  
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
        
        # Get last step of pipeline (works for both LDA and NMF)
        model = list(pipeline.named_steps.values())[-1]
        
        # Top words
        top_indices = model.components_[topic_index].argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices]
        st.subheader("🔍 Top Words for this Topic")
        st.write(", ".join(top_words))
        
        # Bar chart of top words
        top_values = model.components_[topic_index][top_indices]
        fig, ax = plt.subplots()
        ax.barh(top_words[::-1], top_values[::-1], color="skyblue")
        ax.set_xlabel("Word Importance")
        ax.set_title(f"Top Words in Topic {topic_index + 1}")
        st.pyplot(fig)

import os
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import docx
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
import re

# ----------------------
# FORCE CPU & DISABLE TF
# ----------------------
os.environ["USE_TF"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----------------------
# MODEL LOADING (Offline)
# ----------------------
@st.cache_resource
def load_local_model():
    """
    Load a HuggingFace/SentenceTransformer model from a local folder offline.
    The folder must contain a full SentenceTransformer model (config.json, modules.json, pytorch_model.bin).
    """
    model_path = "Model/Resume_Screening"  # folder path, not .pt file
    model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    return model

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# ----------------------
# STATIC SKILLS
# ----------------------
SKILLS = [
    "python", "java", "sql", "excel", "tableau", "power bi",
    "machine learning", "nlp", "deep learning", "data analysis",
    "tensorflow", "pytorch", "communication", "leadership", "statistics"
]

# ----------------------
# FILE PARSING
# ----------------------
def extract_text(uploaded):
    if uploaded.type == "application/pdf":
        with pdfplumber.open(uploaded) as pdf:
            return "\n".join([page.extract_text() or '' for page in pdf.pages])
    if uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded)
        return "\n".join(p.text for p in doc.paragraphs)
    return uploaded.read().decode("utf-8", errors="ignore")

# ----------------------
# SKILL EXTRACTION
# ----------------------
def detect_skills(text):
    t = text.lower()
    return [s for s in SKILLS if s in t]

# ----------------------
# BATCH ENCODING
# ----------------------
def encode_in_batches(model, sentences, batch_size=16):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
        embeddings.append(emb)
    return torch.cat(embeddings)

# ----------------------
# BEST RESUME SNIPPET
# ----------------------
def best_snippet(model, nlp_model, resume_text, job_text):
    sentences = [s.text for s in nlp_model(resume_text).sents if len(s.text.strip()) > 25]
    if not sentences:
        return "No strong content detected in resume."
    sent_emb = encode_in_batches(model, sentences)
    job_emb = model.encode([job_text], convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(job_emb, sent_emb)[0]
    idx = int(torch.argmax(scores).item())
    return sentences[idx]

# ----------------------
# STREAMLIT UI
# ----------------------
st.set_page_config(page_title="Resume Screening", layout="wide")
st.title("📄 Smart Resume Matcher")
st.write("Drop a resume. Paste a job description. Get a real match score.")

model = load_local_model()
nlp_model = load_spacy_model()

resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
job_desc = st.text_area("Paste Job Description", height=200)

if resume_file and job_desc.strip():
    with st.spinner("Processing resume..."):
        text = extract_text(resume_file)
        skills = detect_skills(text)

        sentences = re.split(r'(?<=[.!?])\s+', text)
        res_emb = encode_in_batches(model, sentences)
        job_emb = model.encode([job_desc], convert_to_tensor=True, normalize_embeddings=True)

        sim = util.cos_sim(res_emb.mean(dim=0, keepdim=True), job_emb)[0][0].item()
        snippet = best_snippet(model, nlp_model, text, job_desc)

    st.success("Done.")
    st.metric("Match Score", f"{round(sim * 100, 2)}%")

    st.subheader("Relevant Resume Line")
    st.info(snippet)

    st.subheader("Detected Skills")
    st.write(", ".join(skills) if skills else "None detected")

else:
    st.info("Upload a resume and paste a job description to start.")
import streamlit as st
import fitz  # PyMuPDF
import docx
import pytesseract
from pdf2image import convert_from_bytes
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from rake_nltk import Rake
import spacy
import nltk
import ssl
import pandas as pd
import json
import base64
import os

# Fix SSL for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Ensure required NLTK resources
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.set_page_config(page_title="Automated Metadata Generator", layout="wide")
st.title("📄 Automated Metadata Generator")
st.write("Upload a PDF, DOCX, or TXT file to extract keywords, entities, and summary.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

# -------- File Readers --------
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    if text.strip():
        return text
    else:
        file.seek(0)
        images = convert_from_bytes(file.read())
        return "\n".join([pytesseract.image_to_string(img) for img in images])

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file):
    return file.read().decode("utf-8")

# -------- Extractors --------
def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:10]

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=3)
    return " ".join(str(sentence) for sentence in summary)

# -------- Downloads --------
def download_button(data, filename, label, mime):
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{label}</a>'
    return href

# -------- Main Logic --------
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.subheader("📌 Extracted Text (Truncated):")
    st.text_area("Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)

    keywords = extract_keywords(text)
    entities = extract_entities(text)
    summary = extract_summary(text)

    st.subheader("🔍 Keywords:")
    st.write(keywords)

    st.subheader("🧠 Named Entities:")
    if entities:
        for ent, label in entities:
            st.write(f"**{ent}** – {label}")
    else:
        st.write("No named entities found.")

    st.subheader("📝 Summary:")
    st.write(summary)

    st.success("✅ Metadata extraction completed successfully.")

    # -------- Export Metadata --------
    metadata = {
        "keywords": keywords,
        "entities": [{"text": ent, "label": label} for ent, label in entities],
        "summary": summary
    }

    st.subheader("⬇️ Download Metadata")
    json_data = json.dumps(metadata, indent=2)
    csv_data = pd.DataFrame(metadata["entities"]).to_csv(index=False) if metadata["entities"] else "text,label\n"

    st.markdown(download_button(json_data, "metadata.json", "📥 Download JSON", "application/json"), unsafe_allow_html=True)
    st.markdown(download_button(csv_data, "entities.csv", "📥 Download Entities CSV", "text/csv"), unsafe_allow_html=True)

else:
    st.info("👆 Please upload a document to begin.")





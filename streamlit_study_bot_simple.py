#!/usr/bin/env python3
"""
Regional Language Study Bot - Streamlit App (Simplified for Cloud Deployment)
"""

import streamlit as st
import os
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# IMPORTANT: Set page config first
st.set_page_config(
    page_title="Regional Language Study Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"\'')
                    os.environ[key] = value

load_env()

# SQLite fix for Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Core imports with graceful fallbacks
FEATURES = {
    'groq': False,
    'documents': False,
    'vector_db': False,
    'translation': False
}

# Try GROQ imports
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import SecretStr
    FEATURES['groq'] = True
except ImportError:
    st.error("❌ GROQ/LangChain not available. Please check requirements.")

# Try document processing imports
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    FEATURES['documents'] = True
except ImportError:
    st.warning("⚠️ Document processing not available.")

# Try ChromaDB imports
try:
    import chromadb
    from chromadb.utils import embedding_functions
    FEATURES['vector_db'] = True
except ImportError:
    st.warning("⚠️ Vector database not available.")

# Try translation imports
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    FEATURES['translation'] = True
except ImportError:
    st.info("ℹ️ Translation features not available (transformers/torch missing).")

# Language mapping
INDIAN_LANGUAGES = {
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym"
}

# Initialize session state
def init_session_state():
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'quiz' not in st.session_state:
        st.session_state.quiz = ""

# Load GROQ LLM
@st.cache_resource
def load_groq_llm():
    if not FEATURES['groq']:
        return None
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found!")
            return None
        llm = ChatGroq(
            api_key=SecretStr(groq_api_key),
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load GROQ: {e}")
        return None

# Extract text from document
def extract_text_from_document(uploaded_file) -> str:
    if not FEATURES['documents']:
        return "Document processing not available. Please install required packages."
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])
        elif uploaded_file.name.endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(tmp_path)
            doc = loader.load()
            text = "\n".join([page.page_content for page in doc])
        else:
            text = uploaded_file.getvalue().decode('utf-8')
        
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Generate summary
def generate_summary(text: str) -> str:
    llm = load_groq_llm()
    if not llm:
        return "Summary generation not available (GROQ not configured)"
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a comprehensive summary of the following text:"),
            ("user", "{text}")
        ])
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content
    except Exception as e:
        return f"Error generating summary: {e}"

# Generate quiz
def generate_quiz(text: str) -> str:
    llm = load_groq_llm()
    if not llm:
        return "Quiz generation not available (GROQ not configured)"
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create 5 multiple choice questions from the text. 
            Format as JSON: {"questions": [{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct_answer": "A", "explanation": "..."}]}"""),
            ("user", "{text}")
        ])
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content
    except Exception as e:
        return f"Error generating quiz: {e}"

# Main app
def main():
    init_session_state()
    
    st.title("📚 Regional Language Study Bot")
    st.markdown("Upload documents → Generate summaries & quizzes")
    
    # Show available features
    with st.sidebar:
        st.subheader("🔧 Available Features")
        for feature, available in FEATURES.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {feature.title()}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['pdf', 'docx', 'doc', 'txt']
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            text = extract_text_from_document(uploaded_file)
            st.session_state.extracted_text = text
        
        if text:
            st.success(f"Extracted {len(text)} characters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(text)
                        st.session_state.summary = summary
            
            with col2:
                if st.button("Generate Quiz"):
                    with st.spinner("Generating quiz..."):
                        quiz = generate_quiz(text)
                        st.session_state.quiz = quiz
    
    # Display results
    if st.session_state.extracted_text:
        st.subheader("📄 Extracted Text")
        st.text_area("Content", st.session_state.extracted_text, height=200)
    
    if st.session_state.summary:
        st.subheader("📋 Summary")
        st.write(st.session_state.summary)
    
    if st.session_state.quiz:
        st.subheader("❓ Quiz")
        st.write(st.session_state.quiz)

if __name__ == "__main__":
    main()

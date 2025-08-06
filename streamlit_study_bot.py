#!/usr/bin/env python3
"""
Regional Language Study Bot - Streamlit App
Complete workflow: PDF/DOC extraction -> Summary -> Quiz -> Translation to Indian Languages
"""

import streamlit as st
import os
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Imports after env loading
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    st.error(f"Required packages not installed: {e}")
    st.stop()

# Streamlit page config
st.set_page_config(
    page_title="Regional Language Study Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language mapping for NLLB-200 model
INDIAN_LANGUAGES = {
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng", 
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Urdu": "urd_Arab",
    "Nepali": "npi_Deva",
    "Sanskrit": "san_Deva",
    "Kashmiri": "kas_Arab",
    "Sindhi": "snd_Arab",
    "Konkani": "kok_Deva"
}

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'quiz' not in st.session_state:
        st.session_state.quiz = ""
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    if 'translated_summary' not in st.session_state:
        st.session_state.translated_summary = ""
    if 'translated_quiz' not in st.session_state:
        st.session_state.translated_quiz = ""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

# Load models (cached)
@st.cache_resource
def load_translation_model():
    """Load Facebook NLLB-200 translation model"""
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load translation model: {e}")
        return None, None

@st.cache_resource
def load_groq_llm():
    """Load Groq LLM"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment!")
            return None
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load Groq LLM: {e}")
        return None

@st.cache_resource
def setup_chromadb():
    """Setup ChromaDB for vector storage"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(
            name="document_chunks",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        return client, collection
    except Exception as e:
        st.error(f"Failed to setup ChromaDB: {e}")
        return None, None

# Core functions
def extract_text_from_document(uploaded_file) -> str:
    """Extract text from uploaded document"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])
        elif uploaded_file.name.endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(tmp_path)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.getvalue().decode('utf-8')
        else:
            raise ValueError("Unsupported file format")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return text.strip()
    
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def split_text_into_chunks(text: str, chunk_size: int = 2000) -> List[str]:
    """Split text into manageable chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return [text]

def generate_summary_with_groq(text: str) -> str:
    """Generate summary using Groq LLM"""
    try:
        llm = load_groq_llm()
        if not llm:
            return "Error: Could not load Groq LLM"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating comprehensive educational summaries. 
            Create a detailed summary of the given text that captures:
            - Key concepts and main ideas
            - Important facts and details
            - Learning objectives
            - Critical points for studying
            
            Make it well-structured and easy to understand for students."""),
            ("user", "Text to summarize:\n\n{text}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content
        
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

def generate_quiz_with_groq(text: str) -> str:
    """Generate quiz using Groq LLM"""
    try:
        llm = load_groq_llm()
        if not llm:
            return "Error: Could not load Groq LLM"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert quiz generator. Create exactly 5 high-quality multiple-choice questions based on the given text.
            
            Format your response as valid JSON:
            {{
                "questions": [
                    {{
                        "question": "Question text",
                        "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
                        "correct_answer": "A",
                        "explanation": "Brief explanation of why this answer is correct"
                    }}
                ]
            }}
            
            Make sure questions test understanding, not just memorization."""),
            ("user", "Text to create quiz from:\n\n{text}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content
        
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return f"Error generating quiz: {str(e)}"

def translate_chunk_nllb(chunk: str, target_language_code: str, tokenizer, model) -> str:
    """Translate a single chunk using NLLB-200 model"""
    try:
        inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            translated_tokens = model.generate(
                inputs, 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language_code),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return f"[Translation Error: {str(e)}]"

def translate_text_nllb_parallel(text: str, target_language_code: str, chunk_size: int = 400) -> str:
    """Translate text using NLLB-200 model with parallel chunk processing"""
    try:
        tokenizer, model = load_translation_model()
        if not tokenizer or not model:
            return f"Translation Error: Could not load model"
        
        # Split text into chunks for parallel processing
        if len(text) <= chunk_size:
            # Small text, translate directly
            return translate_chunk_nllb(text, target_language_code, tokenizer, model)
        
        # Large text, split into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Show progress for chunk translation
        progress_container = st.container()
        with progress_container:
            st.info(f"📦 Translating {len(chunks)} chunks in parallel...")
            
        # Create progress bar for chunks
        chunk_progress = st.progress(0)
        translated_chunks = [""] * len(chunks)  # Pre-allocate to maintain order
        
        # Use ThreadPoolExecutor for true parallel processing
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            # Submit all chunks for translation
            future_to_index = {
                executor.submit(translate_chunk_nllb, chunk, target_language_code, tokenizer, model): i 
                for i, chunk in enumerate(chunks)
            }
            
            completed = 0
            for future in as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    translated_chunk = future.result()
                    translated_chunks[chunk_index] = translated_chunk
                    completed += 1
                    
                    # Update progress
                    progress = completed / len(chunks)
                    chunk_progress.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error translating chunk {chunk_index}: {e}")
                    translated_chunks[chunk_index] = f"[Error: {str(e)}]"
                    completed += 1
                    chunk_progress.progress(completed / len(chunks))
        
        # Complete progress
        chunk_progress.progress(1.0)
        progress_container.empty()  # Remove progress info
        
        # Join all translated chunks in correct order
        full_translation = " ".join(translated_chunks)
        return full_translation
        
    except Exception as e:
        st.error(f"Translation error: {e}")
        return f"Translation Error: {str(e)}"

def translate_text_nllb(text: str, target_language_code: str) -> str:
    """Main translation function with parallel chunk processing"""
    return translate_text_nllb_parallel(text, target_language_code)

def store_chunks_in_chromadb(chunks: List[str], document_name: str):
    """Store text chunks in ChromaDB"""
    try:
        client, collection = setup_chromadb()
        if not client or not collection:
            return False
        
        # Prepare documents for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "document_name": document_name,
                "chunk_index": i,
                "chunk_size": len(chunk)
            })
            ids.append(f"{document_name}_chunk_{i}")
        
        # Store in ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error storing chunks: {e}")
        return False

# Main Streamlit App
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("📚 Regional Language Study Bot")
    st.markdown("""
    **Transform your study materials into your native language!**
    
    Upload documents (PDF, DOC, TXT) → Extract text → Generate summary & quiz → Translate to Indian regional languages
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        st.subheader("🤖 Model Status")
        
        # Check Groq LLM
        groq_llm = load_groq_llm()
        if groq_llm:
            st.success("✅ Groq LLM Ready")
        else:
            st.error("❌ Groq LLM Failed")
            st.stop()
        
        # Check Translation Model
        tokenizer, model = load_translation_model()
        if tokenizer and model:
            st.success("✅ NLLB-200 Translation Ready")
        else:
            st.error("❌ Translation Model Failed")
            st.stop()
        
        # ChromaDB status
        client, collection = setup_chromadb()
        if client and collection:
            st.success("✅ ChromaDB Ready")
        else:
            st.warning("⚠️ ChromaDB Optional")
        
        st.divider()
        
        # Language selection
        st.subheader("🌐 Select Target Language")
        selected_language = st.selectbox(
            "Choose your preferred Indian language:",
            options=list(INDIAN_LANGUAGES.keys()),
            index=0
        )
        
        st.info(f"Selected: **{selected_language}**")
        
        st.divider()
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        chunk_size = st.slider("Text Chunk Size", 1000, 5000, 2000, 500)
        translation_chunk_size = st.slider("Translation Chunk Size", 200, 800, 400, 100)
        store_in_db = st.checkbox("Store chunks in ChromaDB", value=True)
        parallel_translation = st.checkbox("Parallel Translation", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'doc', 'docx', 'txt'],
            help="Upload your study material (PDF, DOC, DOCX, or TXT)"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # File info
            file_size = len(uploaded_file.getvalue())
            st.info(f"File size: {file_size:,} bytes")
            
            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                process_document(uploaded_file, selected_language, chunk_size, translation_chunk_size, store_in_db, parallel_translation)
    
    with col2:
        st.header("💬 User Query")
        
        user_query = st.text_area(
            "Ask a question about your document:",
            placeholder="What are the main topics covered in this document?",
            height=100
        )
        
        if user_query and st.session_state.processing_complete:
            if st.button("🔍 Process Query"):
                process_user_query(user_query, selected_language)
    
    # Results section
    if st.session_state.processing_complete:
        display_results(selected_language)

def process_document(uploaded_file, target_language, chunk_size, translation_chunk_size, store_in_db, parallel_translation):
    """Process the uploaded document with configurable translation settings"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Extract text
        status_text.text("📄 Extracting text from document...")
        progress_bar.progress(10)
        
        extracted_text = extract_text_from_document(uploaded_file)
        if not extracted_text:
            st.error("Failed to extract text from document")
            return
        
        st.session_state.extracted_text = extracted_text
        
        # Step 2: Split into chunks
        status_text.text("✂️ Splitting text into chunks...")
        progress_bar.progress(20)
        
        chunks = split_text_into_chunks(extracted_text, chunk_size)
        
        # Step 3: Store in ChromaDB (optional)
        if store_in_db:
            status_text.text("💾 Storing chunks in vector database...")
            progress_bar.progress(30)
            store_chunks_in_chromadb(chunks, uploaded_file.name)
        
        # Step 4: Generate summary
        status_text.text("📝 Generating summary with Groq LLM...")
        progress_bar.progress(40)
        
        # Use first few chunks for summary to avoid token limits
        summary_text = " ".join(chunks[:3])  # First 3 chunks
        summary = generate_summary_with_groq(summary_text)
        st.session_state.summary = summary
        
        # Step 5: Generate quiz
        status_text.text("❓ Generating quiz with Groq LLM...")
        progress_bar.progress(60)
        
        quiz = generate_quiz_with_groq(summary)
        st.session_state.quiz = quiz
        
        # Step 6: Translate content with parallel processing
        target_lang_code = INDIAN_LANGUAGES[target_language]
        
        if parallel_translation:
            status_text.text(f"🚀 Translating full text to {target_language} (Parallel Processing)...")
        else:
            status_text.text(f"🌐 Translating full text to {target_language}...")
        progress_bar.progress(70)
        
        # Translate FULL extracted text with configurable chunk size
        if parallel_translation:
            translated_text = translate_text_nllb_parallel(extracted_text, target_lang_code, translation_chunk_size)
        else:
            translated_text = translate_text_nllb(extracted_text, target_lang_code)
        st.session_state.translated_text = translated_text
        
        progress_bar.progress(80)
        
        # Translate summary
        translated_summary = translate_text_nllb(summary, target_lang_code)
        st.session_state.translated_summary = translated_summary
        
        progress_bar.progress(90)
        
        # Translate quiz
        translated_quiz = translate_text_nllb(quiz, target_lang_code)
        st.session_state.translated_quiz = translated_quiz
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        st.session_state.processing_complete = True
        
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during processing: {e}")

def process_user_query(query, target_language):
    """Process user query and translate response"""
    try:
        with st.spinner("Processing your query..."):
            # Generate response using Groq
            llm = load_groq_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question based on the context of the uploaded document."),
                ("user", f"Document context: {st.session_state.extracted_text[:2000]}...\n\nUser question: {query}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"text": query})
            
            # Translate response
            target_lang_code = INDIAN_LANGUAGES[target_language]
            translated_response = translate_text_nllb(response.content, target_lang_code)
            
            # Display results
            st.subheader("🔍 Query Response")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**English Response:**")
                st.write(response.content)
            
            with col2:
                st.write(f"**{target_language} Response:**")
                st.write(translated_response)
                
    except Exception as e:
        st.error(f"Error processing query: {e}")

def display_results(target_language):
    """Display processing results"""
    st.header("📋 Results")
    
    # Create tabs for different content
    tab1, tab2, tab3 = st.tabs(["📄 Summary", "❓ Quiz", "🔤 Original Text"])
    
    with tab1:
        st.subheader("📝 Summary")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**English Summary:**")
            st.write(st.session_state.summary)
        
        with col2:
            st.write(f"**{target_language} Summary:**")
            st.write(st.session_state.translated_summary)
    
    with tab2:
        st.subheader("❓ Quiz")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**English Quiz:**")
            try:
                quiz_data = json.loads(st.session_state.quiz)
                for i, q in enumerate(quiz_data.get("questions", []), 1):
                    st.write(f"**Q{i}: {q['question']}**")
                    for option in q['options']:
                        st.write(f"  {option}")
                    st.write(f"✅ Correct: {q['correct_answer']}")
                    st.write(f"💡 {q['explanation']}")
                    st.write("---")
            except:
                st.write(st.session_state.quiz)
        
        with col2:
            st.write(f"**{target_language} Quiz:**")
            st.write(st.session_state.translated_quiz)
    
    with tab3:
        st.subheader("📄 Full Document Text")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Original Text:**")
            # Show full text but with scrollable area
            st.text_area("", st.session_state.extracted_text, height=400, disabled=True, key="original_full")
        
        with col2:
            st.write(f"**{target_language} Translation:**")
            # Show full translated text
            st.text_area("", st.session_state.translated_text, height=400, disabled=True, key="translated_full")
    
    # Download options
    st.subheader("💾 Download Results")
    
    # Show processing stats
    if st.session_state.extracted_text and st.session_state.translated_text:
        original_length = len(st.session_state.extracted_text)
        translated_length = len(st.session_state.translated_text)
        st.info(f"📊 **Processing Stats**: Original: {original_length:,} chars → Translated: {translated_length:,} chars")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📥 Download Summary"):
            download_content = f"SUMMARY\n\nEnglish:\n{st.session_state.summary}\n\n{target_language}:\n{st.session_state.translated_summary}"
            st.download_button("Download", download_content, "summary.txt", "text/plain")
    
    with col2:
        if st.button("📥 Download Quiz"):
            download_content = f"QUIZ\n\nEnglish:\n{st.session_state.quiz}\n\n{target_language}:\n{st.session_state.translated_quiz}"
            st.download_button("Download", download_content, "quiz.txt", "text/plain")
    
    with col3:
        if st.button("📥 Download All"):
            download_content = f"""COMPLETE STUDY MATERIAL
            
SUMMARY
English: {st.session_state.summary}
{target_language}: {st.session_state.translated_summary}

QUIZ
English: {st.session_state.quiz}
{target_language}: {st.session_state.translated_quiz}

ORIGINAL TEXT (First Chunk)
English: {st.session_state.extracted_text[:2000]}
{target_language}: {st.session_state.translated_text}
"""
            st.download_button("Download", download_content, "complete_study_material.txt", "text/plain")

if __name__ == "__main__":
    main()

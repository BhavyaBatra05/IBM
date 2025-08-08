#!/usr/bin/env python3
"""
Regional Language Study Bot - Streamlit App
Complete workflow: PDF/DOC extraction -> Summary -> Quiz -> Translation to Indian Languages
"""

# Import streamlit first
import streamlit as st

# IMPORTANT: Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Regional Language Study Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SQLite compatibility fix for ChromaDB on deployment platforms
import sys
import os
import time
import socket
import subprocess
from contextlib import contextmanager

# Function to wait for supervisor connection on cloud deployment
def wait_for_supervisor(max_retries=5, retry_delay=2):
    """Wait for the supervisor socket to be available (for cloud deployments)"""
    supervisor_path = '/mount/admin/.supervisor.sock'
    
    # Only attempt connection on Linux/Unix platforms (where Streamlit deploys)
    if not os.path.exists(supervisor_path) or sys.platform.startswith('win'):
        return True
        
    for i in range(max_retries):
        try:
            # Check if socket file exists and platform supports Unix sockets
            if hasattr(socket, 'AF_UNIX'):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(3)
                s.connect(supervisor_path)
                s.close()
                print("✅ Successfully connected to supervisor socket")
                return True
        except (socket.error, FileNotFoundError):
            print(f"⏳ Waiting for supervisor socket (attempt {i+1}/{max_retries})...")
            
        time.sleep(retry_delay)
    
    print("⚠️ Could not connect to supervisor socket, but continuing anyway...")
    return False

# Add retry logic for startup resilience on cloud platforms with improved nested import handling
def import_with_retry(module_name, max_retries=5, retry_delay=3, timeout=60):
    """Try to import a module with retries for cloud deployment resilience"""
    start_time = time.time()
    for i in range(max_retries):
        # Check timeout to prevent hanging during deployment
        if time.time() - start_time > timeout:
            print(f"Import timeout after {timeout}s for {module_name}")
            return False
            
        try:
            if module_name == 'pysqlite3':
                __import__(module_name)
                sys.modules['sqlite3'] = sys.modules.pop(module_name)
                return True
            elif '.' in module_name:
                # Handle nested imports like 'transformers.pipeline'
                parts = module_name.split('.')
                base_module = __import__(parts[0])
                
                # Walk through the import path
                module = base_module
                for part in parts[1:]:
                    module = getattr(module, part)
                return module
            else:
                return __import__(module_name)
        except (ImportError, AttributeError) as e:
            if i < max_retries - 1:
                # Exponential backoff for more resilience
                current_delay = retry_delay * (2**i)
                print(f"Retry {i+1}/{max_retries} importing {module_name}. Waiting {current_delay}s: {e}")
                time.sleep(current_delay)
            else:
                print(f"Failed to import {module_name} after {max_retries} attempts: {e}")
                raise
    return None

# Try importing pysqlite3 with retries
try:
    import_with_retry('pysqlite3')
except ImportError:
    print("SQLite compatibility layer not available, using system SQLite")

import streamlit as st
import tempfile
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
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

# Check if we're in a cloud deployment and wait for supervisor if needed
is_cloud_deployment = os.path.exists('/mount/admin')
if is_cloud_deployment:
    print("📦 Cloud deployment detected, checking for supervisor service...")
    wait_for_supervisor(max_retries=10, retry_delay=3)

# Load environment variables
load_env()

# Imports after env loading
try:
    # Core imports - always available
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from pydantic import SecretStr
    import requests
    
    # ChromaDB for vector storage with retries
    CHROMADB_AVAILABLE = False
    try:
        # Try importing with retry logic for more stability
        import chromadb
        from chromadb.utils import embedding_functions
        CHROMADB_AVAILABLE = True
        st.success("✅ ChromaDB vector storage available")
    except Exception as e:
        st.warning(f"ChromaDB import issue: {str(e)}")
        # Continue app execution even if ChromaDB fails
        st.info("💡 App will use in-memory storage instead")
        CHROMADB_AVAILABLE = False
    
    # Translation models using Hugging Face Pipeline with retries
    NLLB_TRANSLATION_AVAILABLE = False
    try:
        # Direct import with proper error handling
        import transformers
        from transformers import pipeline
        torch = import_with_retry('torch')
        sentencepiece = import_with_retry('sentencepiece')
        NLLB_TRANSLATION_AVAILABLE = True
        st.success("🚀 NLLB-200 translation pipeline available")
    except Exception as e:
        print(f"Translation import issue: {e}")
        print("💡 App will use simplified translation display")
        NLLB_TRANSLATION_AVAILABLE = False
        
except ImportError as e:
    print(f"Required packages not installed: {e}")
    st.stop()

# Page config already set at the top of the file
# Language mapping setup continues here

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
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = None
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

# Load models (cached)
@st.cache_resource
def load_translation_pipeline():
    """Load Facebook NLLB-200 translation pipeline (deployment-friendly)"""
    if not NLLB_TRANSLATION_AVAILABLE:
        st.info("ℹ️ Translation pipeline not available, using fallback")
        return None
        
    try:
        # Use pipeline API which handles tokenizer and model automatically
        translator = pipeline(
            "translation", 
            model="facebook/nllb-200-distilled-600M",
            device=-1,  # Use CPU for compatibility
            torch_dtype="auto"
        )
        st.success("✅ NLLB-200 translation pipeline loaded")
        return translator
    except Exception as e:
        st.warning(f"Failed to load translation pipeline: {e}")
        return None

@st.cache_resource
def load_groq_llm():
    """Load Groq LLM"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment!")
            return None
        
        llm = ChatGroq(
            api_key=SecretStr(groq_api_key),
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load Groq LLM: {e}")
        return None

@st.cache_resource
def setup_chromadb():
    """Setup ChromaDB for vector storage (if available)"""
    if not CHROMADB_AVAILABLE:
        st.info("📁 Using in-memory document storage (ChromaDB not available)")
        return None
        
    try:
        # Check if chromadb was imported successfully
        if not CHROMADB_AVAILABLE:
            st.warning("ChromaDB not imported, using in-memory storage")
            return None
            
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection with default settings (avoids type issues)
        collection = client.get_or_create_collection(
            name="document_chunks"
        )
        
        st.success("✅ ChromaDB vector storage initialized")
        return collection
    except Exception as e:
        st.warning(f"ChromaDB setup failed, using in-memory storage: {e}")
        return None

def clear_chromadb():
    """Clear ChromaDB data in case of issues"""
    try:
        # First, clear the cached resource to close any open connections
        setup_chromadb.clear()
        
        # Force garbage collection to help close any remaining references
        import gc
        gc.collect()
        
        # Wait a moment for connections to close
        time.sleep(1)
        
        db_path = Path("./chroma_db")
        if db_path.exists():
            try:
                # Try to remove the directory
                shutil.rmtree(db_path)
                st.success("✅ ChromaDB data cleared successfully!")
                st.info("🔄 Please refresh the page to reinitialize ChromaDB")
            except PermissionError:
                # If files are still locked, try to remove individual files
                st.warning("⚠️ Some ChromaDB files are locked. Attempting alternative cleanup...")
                
                # Try to rename the directory first (sometimes this works when delete doesn't)
                backup_path = Path(f"./chroma_db_backup_{int(time.time())}")
                try:
                    db_path.rename(backup_path)
                    st.success("✅ ChromaDB data moved to backup. Will be cleaned up on next restart.")
                    st.info("🔄 Please restart the application to complete cleanup")
                except Exception as rename_error:
                    st.error(f"❌ Could not move ChromaDB files: {rename_error}")
                    st.info("💡 **Solution**: Please stop the Streamlit app (Ctrl+C) and restart it to clear ChromaDB files")
        else:
            st.info("ℹ️ No ChromaDB data found to clear")
            
    except Exception as e:
        st.error(f"❌ Error clearing ChromaDB: {e}")
        st.info("💡 **Manual Solution**: Stop the app (Ctrl+C), delete the 'chroma_db' folder manually, then restart")

def safe_chromadb_reset():
    """Safely reset ChromaDB by stopping the app"""
    st.warning("🔄 **ChromaDB Reset Required**")
    st.info("""
    To properly clear ChromaDB:
    1. **Stop** this Streamlit app (Ctrl+C in terminal)
    2. **Delete** the `chroma_db` folder manually 
    3. **Restart** the app with `streamlit run streamlit_study_bot.py`
    
    This ensures all database connections are properly closed.
    """)
    
    if st.button("🛑 Stop Application", type="secondary"):
        st.success("Stopping application... Please restart manually after deleting chroma_db folder")
        st.stop()

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
        response_text = response.content if hasattr(response, 'content') else str(response)
        return str(response_text)
        
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

def translate_quiz_properly(quiz_json_str: str, target_language_code: str) -> str:
    """Translate quiz while preserving JSON structure"""
    try:
        st.info("🔧 Starting structured quiz translation...")
        
        # First, clean the quiz JSON string
        cleaned_quiz = quiz_json_str.strip()
        
        # Look for JSON content within the string
        if "```json" in cleaned_quiz:
            # Extract JSON from markdown code block
            start_idx = cleaned_quiz.find("```json") + 7
            end_idx = cleaned_quiz.find("```", start_idx)
            if end_idx != -1:
                cleaned_quiz = cleaned_quiz[start_idx:end_idx].strip()
        elif "```" in cleaned_quiz:
            # Extract from generic code block
            start_idx = cleaned_quiz.find("```") + 3
            end_idx = cleaned_quiz.find("```", start_idx)
            if end_idx != -1:
                cleaned_quiz = cleaned_quiz[start_idx:end_idx].strip()
        
        # Try to find JSON object starting with {
        if not cleaned_quiz.startswith("{"):
            json_start = cleaned_quiz.find("{")
            if json_start != -1:
                cleaned_quiz = cleaned_quiz[json_start:]
        
        st.info(f"📝 Parsing English quiz JSON (length: {len(cleaned_quiz)} chars)")
        
        # Parse the original English quiz
        try:
            quiz_data = json.loads(cleaned_quiz)
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse English quiz JSON: {e}")
            st.error("🔍 Raw quiz data preview:")
            st.code(cleaned_quiz[:300])
            return f'{{"error": "Failed to parse English quiz JSON: {str(e)}"}}'
        
        questions = quiz_data.get("questions", [])
        
        if not questions:
            st.warning("⚠️ No questions found in quiz data, falling back to direct translation")
            return translate_text_nllb(quiz_json_str, target_language_code)
        
        st.success(f"✅ Found {len(questions)} questions to translate")
        
        # Create a new quiz data structure for translation
        translated_quiz = {"questions": []}
        
        # Show progress for quiz translation
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_items = len(questions) * 3  # question + options + explanation
        completed = 0
        
        # Translate each question
        for i, question in enumerate(questions):
            progress_text.text(f"🌐 Translating quiz question {i+1}/{len(questions)}...")
            
            # Translate question text
            question_text = question.get("question", "")
            if question_text:
                translated_question = translate_text_nllb(question_text, target_language_code)
            else:
                translated_question = ""
                
            completed += 1
            progress_bar.progress(completed / total_items)
            
            # Translate each option
            translated_options = []
            options = question.get("options", [])
            for option in options:
                # Extract the letter prefix (A., B., etc.) and text
                if '. ' in option and len(option.split('. ', 1)) == 2:
                    prefix, text = option.split('. ', 1)
                    translated_text = translate_text_nllb(text, target_language_code)
                    translated_options.append(f"{prefix}. {translated_text}")
                else:
                    # If no standard prefix, translate the whole option
                    translated_options.append(translate_text_nllb(option, target_language_code))
            
            completed += 1
            progress_bar.progress(completed / total_items)
            
            # Translate explanation
            explanation = question.get("explanation", "")
            translated_explanation = ""
            if explanation:
                translated_explanation = translate_text_nllb(explanation, target_language_code)
            
            completed += 1
            progress_bar.progress(completed / total_items)
            
            # Build translated question object
            translated_question_obj = {
                "question": translated_question,
                "options": translated_options,
                "correct_answer": question.get("correct_answer", "A"),  # Keep the letter (A, B, C, D) unchanged
                "explanation": translated_explanation
            }
            
            translated_quiz["questions"].append(translated_question_obj)
        
        # Complete and cleanup progress indicators
        progress_bar.progress(1.0)
        progress_text.text("✅ Quiz translation completed!")
        
        # Clean up progress indicators after a short delay
        time.sleep(1)
        progress_text.empty()
        progress_bar.empty()
        
        # Return properly formatted JSON
        final_json = json.dumps(translated_quiz, ensure_ascii=False, indent=2)
        st.success(f"🎉 Successfully translated quiz to {target_language_code} ({len(final_json)} chars)")
        
        return final_json
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing original quiz JSON: {e}")
        # Fallback to direct translation (which might break JSON)
        return translate_text_nllb(quiz_json_str, target_language_code)
    except Exception as e:
        st.error(f"Error in structured quiz translation: {e}")
        return translate_text_nllb(quiz_json_str, target_language_code)

def generate_quiz_with_groq(text: str) -> str:
    """Generate quiz using Groq LLM"""
    try:
        llm = load_groq_llm()
        if not llm:
            return "Error: Could not load Groq LLM"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert quiz generator. Create exactly 12 high-quality multiple-choice questions based on the given text.
            
            IMPORTANT: Return ONLY valid JSON without any markdown formatting, explanatory text, or code blocks.
            
            Format your response exactly like this:
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
            
            Make sure questions:
            - Test understanding, not just memorization
            - Cover different aspects of the content
            - Have clear, unambiguous correct answers
            - Include plausible distractors
            - Vary in difficulty from basic to advanced
            
            Return ONLY the JSON object, no other text."""),
            ("user", "Text to create quiz from:\n\n{text}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"text": text})
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean the response to ensure it's proper JSON
        cleaned_response = str(response_text).strip()
        
        # Remove any markdown formatting
        if "```json" in cleaned_response:
            start_idx = cleaned_response.find("```json") + 7
            end_idx = cleaned_response.find("```", start_idx)
            if end_idx != -1:
                cleaned_response = cleaned_response[start_idx:end_idx].strip()
        elif "```" in cleaned_response:
            start_idx = cleaned_response.find("```") + 3
            end_idx = cleaned_response.find("```", start_idx)
            if end_idx != -1:
                cleaned_response = cleaned_response[start_idx:end_idx].strip()
        
        # Find JSON object if there's extra text
        if not cleaned_response.startswith("{"):
            json_start = cleaned_response.find("{")
            if json_start != -1:
                cleaned_response = cleaned_response[json_start:]
        
        # Validate the JSON by trying to parse it
        try:
            json.loads(cleaned_response)
            return cleaned_response
        except json.JSONDecodeError as e:
            st.warning(f"Generated quiz JSON needs cleaning: {e}")
            return cleaned_response  # Return anyway, might work with the display function
        
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return f"Error generating quiz: {str(e)}"

# Removed old translate_chunk_nllb - now using pipeline approach

def translate_text_nllb_pipeline(text: str, target_language_code: str) -> str:
    """Translate text using NLLB-200 pipeline (deployment-friendly)"""
    try:
        translator = load_translation_pipeline()
        if not translator:
            return f"Translation Error: Could not load translation pipeline"
        
        # Split long text into chunks (pipelines have token limits)
        max_length = 400  # Conservative limit for stability
        if len(text) <= max_length:
            # Short text, translate directly
            result = translator(text, src_lang="eng_Latn", tgt_lang=target_language_code)
            return result[0]['translation_text']
        
        # Long text, split into chunks
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        translated_chunks = []
        
        # Show progress
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            result = translator(chunk, src_lang="eng_Latn", tgt_lang=target_language_code)
            translated_chunks.append(result[0]['translation_text'])
            progress.progress((i + 1) / len(chunks))
        
        progress.empty()
        return " ".join(translated_chunks)
        
    except Exception as e:
        return f"Translation Error: {str(e)}"

# Removed Google Translate function - using NLLB pipeline instead

def translate_text_simple_fallback(text: str, target_language: str) -> str:
    """Simple fallback translation using basic text replacement"""
    # This is a very basic fallback - in production you'd want a proper translation service
    language_names = {
        "Hindi": "हिंदी",
        "Bengali": "বাংলা", 
        "Tamil": "தமிழ்",
        "Telugu": "తెలుగు",
        "Marathi": "मराठी",
        "Gujarati": "ગુજરાતી",
        "Kannada": "ಕನ್ನಡ",
        "Malayalam": "മലയാളം",
        "Punjabi": "ਪੰਜਾਬੀ",
        "Odia": "ଓଡ଼ିଆ",
        "Assamese": "অসমীয়া",
        "Urdu": "اردو"
    }
    
    return f"""
📌 **{language_names.get(target_language, target_language)} Translation Available**

Original English Content:
{text}

🔄 Regional language translation will be added in future updates. 
For now, you can use the English content above for study purposes.
"""

def translate_text_nllb(text: str, target_language_code: str) -> str:
    """Main translation function using NLLB pipeline"""
    # Get target language name
    target_language = None
    for lang, code in INDIAN_LANGUAGES.items():
        if code == target_language_code:
            target_language = lang
            break
    
    # Try NLLB pipeline if available
    if NLLB_TRANSLATION_AVAILABLE:
        try:
            st.info("🚀 Using NLLB-200 translation pipeline")
            return translate_text_nllb_pipeline(text, target_language_code)
        except Exception as e:
            st.warning(f"NLLB translation failed: {e}")
    
    # Fallback to simple placeholder
    return translate_text_simple_fallback(text, target_language or "Unknown")

def store_chunks_in_chromadb(chunks: List[str], document_name: str):
    """Store document chunks in ChromaDB or in-memory storage"""
    if not CHROMADB_AVAILABLE:
        # Use in-memory storage as fallback
        if 'document_chunks' not in st.session_state:
            st.session_state.document_chunks = {}
        
        st.session_state.document_chunks[document_name] = chunks
        st.success(f"Stored {len(chunks)} chunks in memory")
        return
        
    try:
        collection = setup_chromadb()
        if collection is None:
            # Fallback to in-memory storage
            if 'document_chunks' not in st.session_state:
                st.session_state.document_chunks = {}
            
            st.session_state.document_chunks[document_name] = chunks
            st.info(f"Stored {len(chunks)} chunks in memory (ChromaDB fallback)")
            return
            
        # Create metadata for each chunk with proper string types
        # ChromaDB requires all metadata values to be strings, numbers, or booleans
        metadatas = [
            {
                "document": str(document_name),  # Ensure string type
                "chunk_id": str(i),            # Ensure string type
                "total_chunks": str(len(chunks)) # Additional metadata as string
            } 
            for i in range(len(chunks))
        ]
        
        # Generate unique IDs for each chunk
        ids = [f"{document_name}_{i}" for i in range(len(chunks))]
        
        # Add documents to collection with proper error handling
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        st.success(f"Stored {len(chunks)} chunks in vector database")
        
    except Exception as e:
        # Fallback to in-memory storage on error
        if 'document_chunks' not in st.session_state:
            st.session_state.document_chunks = {}
        
        st.session_state.document_chunks[document_name] = chunks
        st.warning(f"ChromaDB error, using memory storage: {e}")
        st.info(f"Stored {len(chunks)} chunks in memory")

def debug_quiz_format(quiz_data, title="Quiz Debug"):
    """Debug function to show quiz data format"""
    st.write(f"**{title}:**")
    st.write(f"- Type: {type(quiz_data)}")
    st.write(f"- Length: {len(str(quiz_data)) if quiz_data else 0}")
    
    if isinstance(quiz_data, str):
        # Show first 200 characters
        preview = quiz_data[:200] + "..." if len(quiz_data) > 200 else quiz_data
        st.code(preview)
        
        # Check for common JSON patterns
        has_json_start = "{" in quiz_data
        has_questions = "questions" in quiz_data.lower()
        has_options = "options" in quiz_data.lower()
        
        st.write(f"- Contains '{{': {has_json_start}")
        st.write(f"- Contains 'questions': {has_questions}")
        st.write(f"- Contains 'options': {has_options}")

def display_interactive_quiz(quiz_data, target_language, quiz_id="main"):
    """Display interactive quiz with scoring"""
    try:
        # Handle different quiz data formats
        if isinstance(quiz_data, str):
            # Try to clean and parse JSON
            cleaned_quiz = quiz_data.strip()
            
            # Look for JSON content within the string
            if "```json" in cleaned_quiz:
                # Extract JSON from markdown code block
                start_idx = cleaned_quiz.find("```json") + 7
                end_idx = cleaned_quiz.find("```", start_idx)
                if end_idx != -1:
                    cleaned_quiz = cleaned_quiz[start_idx:end_idx].strip()
            elif "```" in cleaned_quiz:
                # Extract from generic code block
                start_idx = cleaned_quiz.find("```") + 3
                end_idx = cleaned_quiz.find("```", start_idx)
                if end_idx != -1:
                    cleaned_quiz = cleaned_quiz[start_idx:end_idx].strip()
            
            # Try to find JSON object starting with {
            if not cleaned_quiz.startswith("{"):
                json_start = cleaned_quiz.find("{")
                if json_start != -1:
                    cleaned_quiz = cleaned_quiz[json_start:]
            
            # Try to parse JSON
            try:
                quiz_json = json.loads(cleaned_quiz)
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse quiz JSON: {e}")
                st.error("🔍 **Debug Info:**")
                st.code(cleaned_quiz[:500] + "..." if len(cleaned_quiz) > 500 else cleaned_quiz)
                
                # Fallback: try to create a simple quiz display
                st.warning("Displaying quiz as text format instead:")
                st.text_area("Quiz Content", quiz_data, height=300)
                return None
                
        else:
            quiz_json = quiz_data
        
        questions = quiz_json.get("questions", [])
        if not questions:
            st.error("No questions found in quiz data")
            st.info("Quiz data structure:")
            st.json(quiz_json)
            return
        
        st.subheader(f"📝 Interactive Quiz ({len(questions)} Questions)")
        st.info(f"🎯 Complete the quiz to test your understanding. Each correct answer = 1 point!")
        
        # Quiz container with unique form key
        with st.form(f"quiz_form_{quiz_id}"):
            user_answers = {}
            
            for i, question in enumerate(questions):
                st.markdown(f"---")
                st.markdown(f"**Question {i+1}:** {question['question']}")
                
                # Create radio options without the A., B., C., D. prefixes for cleaner display
                clean_options = []
                option_mapping = {}
                
                for option in question['options']:
                    # Extract just the option text after "A. ", "B. ", etc.
                    if '. ' in option:
                        letter = option.split('. ')[0]
                        text = option.split('. ', 1)[1]
                        clean_options.append(text)
                        option_mapping[text] = letter
                    else:
                        clean_options.append(option)
                        option_mapping[option] = option
                
                selected_option = st.radio(
                    f"Select your answer:",
                    clean_options,
                    key=f"question_{i}_{quiz_id}",
                    index=None
                )
                
                if selected_option:
                    user_answers[i] = option_mapping[selected_option]
            
            st.markdown("---")
            submitted = st.form_submit_button("🎯 Submit Quiz", type="primary")
            
            if submitted:
                return calculate_quiz_score(user_answers, questions, target_language, quiz_id)
        
        return None
        
    except Exception as e:
        st.error(f"Error displaying quiz: {e}")
        return None

def calculate_quiz_score(user_answers, questions, target_language, quiz_id="main"):
    """Calculate and display quiz score"""
    try:
        total_questions = len(questions)
        correct_answers = 0
        
        st.subheader(f"📊 Quiz Results ({quiz_id.title()})")
        
        # Calculate score
        for i, question in enumerate(questions):
            user_answer = user_answers.get(i)
            correct_answer = question['correct_answer']
            
            if user_answer == correct_answer:
                correct_answers += 1
        
        # Display overall score
        score_percentage = (correct_answers / total_questions) * 100
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Score", f"{correct_answers}/{total_questions}")
        
        with col2:
            st.metric("Percentage", f"{score_percentage:.1f}%")
        
        with col3:
            if score_percentage >= 80:
                grade = "🏆 Excellent"
            elif score_percentage >= 60:
                grade = "👍 Good"
            elif score_percentage >= 40:
                grade = "📚 Fair"
            else:
                grade = "📖 Needs Study"
            st.metric("Grade", grade)
        
        # Progress bar
        st.progress(score_percentage / 100, text=f"Quiz Performance: {score_percentage:.1f}%")
        
        # Detailed results
        with st.expander("📋 Detailed Results", expanded=True):
            for i, question in enumerate(questions):
                user_answer = user_answers.get(i, "Not answered")
                correct_answer = question['correct_answer']
                is_correct = user_answer == correct_answer
                
                # Status icon
                status_icon = "✅" if is_correct else "❌"
                
                st.markdown(f"{status_icon} **Question {i+1}:** {question['question']}")
                
                if user_answer != "Not answered":
                    # Find the full option text for user's answer
                    user_option_text = "Not found"
                    correct_option_text = "Not found"
                    
                    for option in question['options']:
                        if option.startswith(f"{user_answer}."):
                            user_option_text = option
                        if option.startswith(f"{correct_answer}."):
                            correct_option_text = option
                    
                    if is_correct:
                        st.success(f"Your answer: {user_option_text} ✅")
                    else:
                        st.error(f"Your answer: {user_option_text} ❌")
                        st.info(f"Correct answer: {correct_option_text}")
                else:
                    st.warning("Not answered")
                    correct_option_text = next(
                        (opt for opt in question['options'] if opt.startswith(f"{correct_answer}.")), 
                        "Not found"
                    )
                    st.info(f"Correct answer: {correct_option_text}")
                
                # Explanation
                if 'explanation' in question and question['explanation']:
                    st.info(f"💡 **Explanation:** {question['explanation']}")
                
                st.markdown("---")
        
        # Store score in session state
        st.session_state.quiz_score = {
            'correct': correct_answers,
            'total': total_questions,
            'percentage': score_percentage
        }
        st.session_state.quiz_submitted = True
        
        return {
            'correct': correct_answers,
            'total': total_questions,
            'percentage': score_percentage
        }
        
    except Exception as e:
        st.error(f"Error calculating score: {e}")
        return None

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
        
        # Check Translation Pipeline
        if NLLB_TRANSLATION_AVAILABLE:
            translator = load_translation_pipeline()
            if translator:
                st.success("✅ NLLB-200 Advanced Translation Ready")
            else:
                st.warning("⚠️ NLLB-200 Failed, using fallback")
        else:
            st.error("❌ No Translation Service Available")
            st.info("💡 App will work with limited functionality")
        
        # Storage status
        if CHROMADB_AVAILABLE:
            collection = setup_chromadb()
            if collection:
                st.success("✅ ChromaDB Vector Storage Ready")
            else:
                st.warning("⚠️ ChromaDB failed, using memory storage")
        else:
            st.info("� Using in-memory storage")
            st.info("💡 ChromaDB not available (SQLite compatibility issue)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Clear ChromaDB", help="Try to clear ChromaDB data"):
                    clear_chromadb()
            with col_b:
                if st.button("🔄 Reset Guide", help="Get instructions for manual reset"):
                    safe_chromadb_reset()
        
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
            try:
                store_chunks_in_chromadb(chunks, uploaded_file.name)
            except Exception as e:
                st.warning(f"⚠️ ChromaDB storage failed: {e}")
                st.info("📝 Continuing without vector storage...")
                # Continue processing without ChromaDB
        
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
        
        # Translate FULL extracted text using NLLB pipeline
        translated_text = translate_text_nllb(extracted_text, target_lang_code)
        st.session_state.translated_text = translated_text
        
        progress_bar.progress(80)
        
        # Translate summary
        translated_summary = translate_text_nllb(summary, target_lang_code)
        st.session_state.translated_summary = translated_summary
        
        progress_bar.progress(90)
        
        # Translate quiz using structured translation to preserve JSON format
        status_text.text(f"❓ Translating quiz to {target_language} (preserving structure)...")
        translated_quiz = translate_quiz_properly(quiz, target_lang_code)
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
            if llm is None:
                st.error("Failed to load LLM")
                return
                
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question based on the context of the uploaded document."),
                ("user", f"Document context: {st.session_state.extracted_text[:2000]}...\n\nUser question: {query}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"text": query})
            
            # Translate response
            target_lang_code = INDIAN_LANGUAGES[target_language]
            response_text = response.content if hasattr(response, 'content') else str(response)
            translated_response = translate_text_nllb(str(response_text), target_lang_code)
            
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
        st.subheader("❓ Interactive Quiz")
        
        # Debug toggle
        debug_mode = st.checkbox("🔧 Debug Mode", help="Show quiz data format for troubleshooting")
        
        if debug_mode and st.session_state.quiz:
            with st.expander("🔍 Debug: Quiz Data Format", expanded=True):
                debug_quiz_format(st.session_state.quiz, "Original English Quiz Data")
                
                if st.session_state.translated_quiz:
                    st.write("---")
                    debug_quiz_format(st.session_state.translated_quiz, "Translated Quiz Data")
                    
                    # Try to parse both to show comparison
                    try:
                        english_quiz = json.loads(st.session_state.quiz) if isinstance(st.session_state.quiz, str) else st.session_state.quiz
                        translated_quiz = json.loads(st.session_state.translated_quiz) if isinstance(st.session_state.translated_quiz, str) else st.session_state.translated_quiz
                        
                        st.write("**English Questions:**", len(english_quiz.get("questions", [])))
                        st.write("**Translated Questions:**", len(translated_quiz.get("questions", [])))
                        
                        if len(english_quiz.get("questions", [])) > 0:
                            st.write("**First English Question:**")
                            st.code(str(english_quiz["questions"][0]))
                            
                        if len(translated_quiz.get("questions", [])) > 0:
                            st.write("**First Translated Question:**")
                            st.code(str(translated_quiz["questions"][0]))
                            
                    except Exception as e:
                        st.write(f"Debug parsing error: {e}")
                        
                # Test translation button
                if st.button("🔬 Test Quiz Translation"):
                    if st.session_state.quiz:
                        target_lang_code = INDIAN_LANGUAGES["Hindi"]  # Test with Hindi
                        with st.spinner("Testing quiz translation..."):
                            test_translation = translate_quiz_properly(st.session_state.quiz, target_lang_code)
                            st.write("**Test Translation Result:**")
                            st.code(test_translation[:500] + "..." if len(test_translation) > 500 else test_translation)
        
        # Show English quiz as interactive version
        if st.session_state.quiz:
            st.write("### 🇺🇸 English Interactive Quiz")
            quiz_result = display_interactive_quiz(st.session_state.quiz, target_language, "english")
            
            # Show translated quiz as ALSO interactive
            if st.session_state.translated_quiz:
                st.write(f"### 🌏 {target_language} Interactive Quiz")
                st.info("Take the same quiz in your regional language!")
                
                try:
                    # Try to parse translated quiz and make it interactive too
                    translated_quiz_data = json.loads(st.session_state.translated_quiz)
                    if translated_quiz_data.get("questions"):
                        # Display as interactive quiz
                        quiz_result_translated = display_interactive_quiz(st.session_state.translated_quiz, target_language, "translated")
                    else:
                        raise ValueError("No questions found in translated quiz")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    st.warning(f"Could not create interactive translated quiz: {e}")
                    # Fall back to reference display
                    with st.expander("� Reference: Translated Quiz", expanded=False):
                        st.write(f"**{target_language} Quiz for Reference:**")
                        st.text_area("Translated Quiz (Raw)", st.session_state.translated_quiz, height=300)
                except Exception as e:
                    st.error(f"Error with translated quiz: {e}")
                    with st.expander("📚 Reference: Translated Quiz", expanded=False):
                        st.text_area("Translated Quiz (Raw)", st.session_state.translated_quiz, height=300)
        else:
            st.info("No quiz generated yet. Upload a document and process it first.")
    
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

# 📚 Regional Language Study Bot - Streamlit Edition

A comprehensive study assistant that processes documents and generates summaries, quizzes, and translations in 17+ Indian regional languages.

## 🌟 Features

### Core Workflow
1. **📄 Document Processing**: Extract text from PDF, DOC, DOCX, TXT files
2. **📝 Summary Generation**: Create comprehensive summaries using Groq LLM
3. **❓ Quiz Generation**: Generate 5 multiple-choice questions with explanations
4. **🌐 Translation**: Translate all content to Indian regional languages using Facebook NLLB-200
5. **💾 Vector Storage**: Store document chunks in ChromaDB for efficient retrieval
6. **💬 Query Processing**: Answer user questions about uploaded documents

### Supported Languages
- Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam
- Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit, Kashmiri, Sindhi, Konkani

### AI Models Used
- **Groq LLM** (llama-3.3-70b-versatile): Text extraction, summarization, quiz generation
- **Facebook NLLB-200** (distilled-600M): Translation to Indian regional languages
- **ChromaDB**: Vector storage for document chunks

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Python 3.8+ required
# Virtual environment recommended
python -m venv myvenv
myvenv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 3. Environment Setup
Create `.env` file:
```bash
GROQ_API_KEY="your-groq-api-key-here"
```

### 4. Run Application
```bash
python run_streamlit_bot.py
```

The app will open at: http://localhost:8501

## 📁 Project Structure
```
├── streamlit_study_bot.py      # Main Streamlit application
├── run_streamlit_bot.py        # Startup script
├── requirements_streamlit.txt  # Python dependencies
├── .env                        # Environment variables
├── chroma_db/                  # ChromaDB vector storage (auto-created)
└── README_streamlit.md         # This file
```

## 🎯 How to Use

### 1. Upload Document
- Click "Choose a file" and upload PDF, DOC, DOCX, or TXT
- Supported file types: Academic papers, textbooks, notes, articles

### 2. Select Language
- Choose your preferred Indian regional language from the sidebar
- The app supports 17+ languages with proper script rendering

### 3. Process Document
- Click "🚀 Start Processing"
- Watch real-time progress:
  - Text extraction from document
  - Chunking for efficient processing
  - Summary generation with Groq LLM
  - Quiz creation with explanations
  - Translation to selected language

### 4. Review Results
- **Summary Tab**: Side-by-side English and translated summaries
- **Quiz Tab**: Interactive quiz with correct answers and explanations
- **Original Text Tab**: Compare original and translated text

### 5. Ask Questions
- Use the query box to ask questions about your document
- Get answers in both English and your selected language

### 6. Download Results
- Download individual components (summary, quiz)
- Download complete study package

## ⚙️ Configuration Options

### Sidebar Settings
- **Model Status**: Real-time status of Groq LLM and NLLB-200
- **Language Selection**: Choose from 17+ Indian languages
- **Text Chunk Size**: Adjust document processing chunk size (1000-5000 tokens)
- **Translation Chunk Size**: Configure translation chunk size (200-800 chars)
- **Parallel Translation**: Enable/disable multi-threaded translation
- **ChromaDB Storage**: Enable/disable vector storage

### Advanced Features
- **Parallel Chunk Translation**: Multi-threaded processing for faster translation
- **Full Document Translation**: Complete text translation with progress tracking
- **Configurable Chunk Sizes**: Optimize for different document types and sizes
- **Vector Storage**: Store and retrieve document chunks with ChromaDB
- **Real-time Progress**: Live updates for each translation chunk
- **Multi-format Support**: PDF, DOC, DOCX, TXT files

## 🔧 Technical Details

### Document Processing Pipeline
1. **Text Extraction**: 
   - PDF: PyPDFLoader
   - DOC/DOCX: Docx2txtLoader
   - TXT: Direct UTF-8 decoding

2. **Text Chunking**:
   - RecursiveCharacterTextSplitter
   - Configurable chunk size (default: 2000 tokens)
   - 200 token overlap for context preservation

3. **LLM Processing**:
   - Groq API with llama-3.3-70b-versatile
   - Structured prompts for consistent output
   - Error handling and fallbacks

4. **Translation**:
   - Facebook NLLB-200 distilled model
   - Batch processing for long texts
   - Language-specific tokenization

5. **Vector Storage**:
   - ChromaDB persistent storage
   - Embeddings for semantic search
   - Document metadata preservation

### Performance Optimizations
- **Model Caching**: Models loaded once using @st.cache_resource
- **Parallel Translation**: Multi-threaded chunk processing for faster translation
- **Configurable Chunk Sizes**: Optimize processing for different document sizes
- **Streaming UI**: Real-time progress updates with chunk-by-chunk translation progress
- **Session State**: Persistent results across interactions
- **Full Document Translation**: Complete text translation (not just samples)

## 🛠️ Troubleshooting

### Common Issues

1. **GROQ_API_KEY not found**
   - Ensure `.env` file exists with valid API key
   - Check API key format and quotes

2. **Model loading errors**
   - Verify internet connection for model downloads
   - Check available disk space (models are ~2GB)
   - Restart app if models fail to load

3. **Translation errors**
   - Large texts are automatically chunked
   - Some languages may have limited model support
   - Check input text encoding

4. **Document upload failures**
   - Verify file format (PDF, DOC, DOCX, TXT)
   - Check file size (large files may take longer)
   - Ensure file is not corrupted

### Performance Tips
- Use smaller chunk sizes for faster processing
- Enable ChromaDB storage for repeated document queries
- Close other applications to free up memory for model loading

## 📊 Example Workflow

1. **Upload**: Course textbook PDF (50 pages)
2. **Process**: Extract text → Generate summary → Create quiz → Translate to Hindi
3. **Study**: Review Hindi summary and take quiz
4. **Query**: "What are the key concepts in Chapter 3?"
5. **Result**: Get answer in both English and Hindi

## 🎓 Educational Use Cases

- **Students**: Transform textbooks into native language study materials
- **Researchers**: Summarize academic papers with translated abstracts
- **Teachers**: Create multilingual quizzes from course materials
- **Self-learners**: Convert any document into structured study content

## 🔮 Future Enhancements

- **More Languages**: Expand to other regional languages
- **Audio Output**: Text-to-speech in regional languages
- **Advanced Quizzes**: Multiple question types and difficulty levels
- **Collaborative Features**: Share study materials with others
- **Mobile App**: Native mobile interface

---

**Built with ❤️ using Streamlit, Groq LLM, and Facebook NLLB-200**

For support or contributions, please visit: [GitHub Repository](https://github.com/your-repo)

# Streamlit Cloud Deployment Guide

## Files Required for Successful Deployment

### 1. `requirements_streamlit.txt` (Use this for Streamlit Cloud)
```
streamlit>=1.28.0
pysqlite3-binary>=0.5.0
langchain-groq>=0.1.0
langchain>=0.1.0
langchain-community>=0.0.10
transformers>=4.30.0
torch>=2.0.0,<2.1.0
sentencepiece>=0.1.99
protobuf>=4.21.0
pypdf>=3.17.0
python-docx>=0.8.11
docx2txt>=0.8
chromadb>=0.4.0
aiohttp>=3.8.0
pydantic>=2.0.0
```

### 2. `packages.txt` (System dependencies)
```
libsqlite3-dev
```

### 3. `.streamlit/config.toml` (App configuration)
```toml
[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[runner]
magicEnabled = false
```

## Environment Variables for Streamlit Cloud

In your Streamlit Cloud app settings, add:
- `GROQ_API_KEY`: Your Groq API key

## Deployment Steps

1. **Push to GitHub**: Make sure all files are committed to your repository
2. **Deploy on Streamlit Cloud**: 
   - Go to https://share.streamlit.io/
   - Connect your GitHub repository
   - Set main file as `streamlit_study_bot.py`
   - Add environment variables in the settings
3. **Monitor Deployment**: Check logs for any remaining issues

## Common Issues and Solutions

### SQLite Version Error
- **Problem**: `RuntimeError: Your system has an unsupported version of sqlite3`
- **Solution**: Added `pysqlite3-binary` to requirements and SQLite import fix in code

### Memory Issues with Transformers
- **Problem**: Models too large for Streamlit Cloud
- **Solution**: Using CPU-only torch version and optimized model loading

### ChromaDB Persistence Issues
- **Problem**: Database connection errors
- **Solution**: Added fallback to in-memory ChromaDB if persistent storage fails

## Testing Locally

Before deploying, test with Streamlit Cloud requirements:
```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_study_bot.py
```

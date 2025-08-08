# 🚀 Streamlit Cloud Deployment Instructions

## 📋 Quick Deploy Guide

### **Option 1: Direct Deploy (Recommended)**

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"**
4. **Connect your GitHub** account
5. **Select your forked repository**
6. **Set main file**: `streamlit_study_bot.py`
7. **Add your environment variables**:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
8. **Click Deploy!**

### **⚙️ Environment Setup**

The app includes automatic fallback for translation services:
- 🚀 **Primary**: NLLB-200 advanced model (if available)
- 🌐 **Fallback**: Google Translate service
- 📝 **Final Fallback**: Basic placeholder text

### **🔧 Deployment Configuration**

Files for Streamlit Cloud:
- `requirements.txt` - Python dependencies (deployment-friendly)
- `packages.txt` - System dependencies
- `.env` - Environment variables (create this with your API key)

### **🛠️ Troubleshooting**

**If deployment fails:**

1. **Check requirements.txt** - Uses lightweight dependencies
2. **Verify API keys** - GROQ_API_KEY must be set
3. **Review logs** - Check Streamlit Cloud deployment logs
4. **Test locally** - Run `streamlit run streamlit_study_bot.py`

**Common Issues:**
- ❌ **sentencepiece build errors** - Fixed by using Google Translate fallback
- ❌ **CUDA/PyTorch issues** - App works without advanced models
- ❌ **ChromaDB errors** - Includes reset functionality

### **🌟 Features Available on Deployment**

✅ Document upload (PDF, DOC, TXT)
✅ Text extraction and summarization
✅ Quiz generation with Groq LLM
✅ Translation to Indian languages
✅ Interactive quiz interface
✅ ChromaDB vector storage (optional)
✅ Responsive web interface

### **📱 Usage After Deploy**

1. **Upload** your study document
2. **Select** target language (Hindi, Bengali, Tamil, etc.)
3. **Process** document for summary and quiz
4. **Take** interactive quizzes in both English and regional language
5. **Download** all results

---

## 🔗 Deploy Now

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

**Repository**: `https://github.com/YOUR_USERNAME/IBM`
**Main file**: `streamlit_study_bot.py`

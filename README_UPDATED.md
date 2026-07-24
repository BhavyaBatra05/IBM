# 📚 Regional Language Study Bot

> **An AI-powered multilingual document study assistant that transforms academic documents into personalized learning material with summaries, quizzes, document Q&A, and translations into 16+ Indian languages.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red?style=for-the-badge&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge)
![Azure AI Translator](https://img.shields.io/badge/Azure-AI%20Translator-0078D4?style=for-the-badge&logo=microsoftazure)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Container-blue?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-orange?style=for-the-badge&logo=amazonaws)

---

# 🌟 Overview

Regional Language Study Bot is an AI-powered educational platform that converts study material into easy-to-understand learning resources.

The application allows users to upload documents, automatically extracts the content, generates AI-powered summaries, creates multiple-choice quizzes, answers document-specific questions using Retrieval-Augmented Generation (RAG), and translates all generated content into multiple Indian regional languages.

The project combines modern Large Language Models with cloud-based translation services to provide an efficient and scalable multilingual learning experience.

---

# ✨ Features

## 📄 Intelligent Document Processing

Supports:

- PDF
- DOCX
- TXT

Automatically extracts and processes text for AI analysis.

---

## 📝 AI Summary Generation

Generate concise and structured summaries using:

- Groq Llama-3.3-70B

The summaries preserve important concepts while reducing reading time.

---

## ❓ Automatic Quiz Generation

Generate AI-powered multiple-choice questions including:

- Question
- Four Options
- Correct Answer
- Explanation

Useful for revision and self-assessment.

---

## 💬 Document Question Answering (RAG)

Ask questions directly about the uploaded document.

Powered by:

- LangChain
- ChromaDB
- Groq LLM

The system retrieves relevant chunks before generating responses.

---

## 🌐 Multilingual Translation

Translate:

- Original Document
- Summary
- Quiz
- AI Responses

using **Microsoft Azure AI Translator**.

Supports **16 Indian regional languages**.

---

## 🐳 Docker Support

The project is fully containerized using Docker for consistent deployment across environments.

---

## ☁️ Cloud Deployment Ready

Designed to run on:

- AWS EC2 Free Tier
- Azure Virtual Machines
- Railway
- Any Docker-compatible server

---

# 🏗 System Architecture

```text
                    +----------------------+
                    |  Upload Document     |
                    +----------+-----------+
                               |
                               v
                     Text Extraction Layer
                               |
                               v
                  Recursive Text Chunking
                               |
               +---------------+---------------+
               |                               |
               |                               |
               v                               v
        Groq LLM                     Azure AI Translator
               |                               |
     +---------+---------+                     |
     |         |         |                     |
     |         |         |                     |
 Summary     Quiz      Q&A                 Translation
     |         |         |                     |
     +---------+---------+---------------------+
                       |
                       v
                 Streamlit Interface
                       |
                       v
                   ChromaDB
```

---

# ⚙ Tech Stack

| Category | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| LLM | Groq (Llama-3.3-70B) |
| Translation | Azure AI Translator |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Document Parsing | PyPDF, python-docx |
| Deployment | Docker |
| Cloud | AWS EC2 |

---

# 🌍 Supported Languages

The application currently supports translation into:

- Hindi
- Bengali
- Tamil
- Telugu
- Marathi
- Gujarati
- Kannada
- Malayalam
- Punjabi
- Odia
- Assamese
- Urdu
- Nepali
- Kashmiri
- Sindhi
- Konkani

Azure AI Translator performs automatic language translation with high accuracy while preserving the original meaning.

---

# 🚀 Key Highlights

✅ AI-powered educational assistant

✅ Retrieval-Augmented Generation (RAG)

✅ Multilingual learning support

✅ Interactive quiz generation

✅ Azure AI Translator integration

✅ Groq LLM integration

✅ ChromaDB vector search

✅ Dockerized deployment

✅ AWS EC2 compatible

✅ Modern cloud-native architecture

---

# 📷 Application Preview

> Add screenshots here.

Example:

```
images/home.png

images/summary.png

images/quiz.png

images/translation.png
```

These screenshots help visitors quickly understand the application.

# 📁 Project Structure

```
Regional-Language-Study-Bot/
│
├── streamlit_study_bot.py          # Main Streamlit application
├── run_streamlit_bot.py            # Startup script
├── requirements.txt                # Project dependencies
├── Dockerfile                      # Docker configuration
├── .env                            # Environment variables (Not committed)
├── .gitignore
│
├── chroma_db/                      # ChromaDB vector database
│
├── assets/                         # Images & screenshots
│
├── README.md
│
└── requirements_streamlit.txt      # (Optional legacy file)
```

---

# ⚡ Installation

## 1. Clone Repository

```bash
git clone https://github.com/<your-username>/regional-language-study-bot.git

cd regional-language-study-bot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a file named

```
.env
```

and add:

```env
GROQ_API_KEY=your_groq_api_key

AZURE_TRANSLATOR_KEY=your_azure_key

AZURE_TRANSLATOR_REGION=centralindia

AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com/
```

---

## Getting API Keys

### Groq API

1. Create an account on Groq Cloud

2. Generate an API Key

3. Copy the key into

```
GROQ_API_KEY
```

---

### Azure AI Translator

1. Create a Translator resource in Azure Portal

2. Choose

```
Pricing Tier: F0 (Free)
```

3. Copy

- Key
- Region
- Endpoint

into the `.env` file.

The free tier provides:

- 2 Million translated characters per month.

---

# ▶ Running the Application

Run:

```bash
streamlit run streamlit_study_bot.py
```

or

```bash
python run_streamlit_bot.py
```

The application will open at

```
http://localhost:8501
```

---

# 🐳 Docker Deployment

## Build Docker Image

```bash
docker build -t ibm-study-bot .
```

---

## Run Docker Container

```bash
docker run --env-file .env -p 8501:8501 ibm-study-bot
```

Then open

```
http://localhost:8501
```

---

# ☁ Deploying on AWS EC2

The application has been designed for deployment on AWS EC2 Free Tier.

---

## Step 1

Launch an Ubuntu EC2 instance.

Recommended:

- Ubuntu 24.04 LTS
- t2.micro (Free Tier)

---

## Step 2

Install Docker

```bash
sudo apt update

sudo apt install docker.io -y
```

---

## Step 3

Enable Docker

```bash
sudo systemctl enable docker

sudo systemctl start docker
```

---

## Step 4

Transfer the project

Clone your GitHub repository

or

Copy the Docker image.

---

## Step 5

Run

```bash
docker run \
--env-file .env \
-p 8501:8501 \
ibm-study-bot
```

---

## Step 6

Open

```
http://<EC2-PUBLIC-IP>:8501
```

Your application is now live.

---

# ⚙ Configuration

The application supports configurable settings including:

- Translation Language
- Text Chunk Size
- ChromaDB Storage
- Quiz Generation
- Summary Generation

These can be modified directly from the Streamlit interface.

---

# 💾 ChromaDB Storage

The uploaded document is converted into embeddings and stored in ChromaDB.

Benefits include:

- Faster document retrieval
- Semantic search
- Accurate question answering
- Persistent vector storage

---

# 🔒 Security

Sensitive information should never be committed.

Ensure `.gitignore` contains:

```gitignore
.env

venv/

__pycache__/

chroma_db/

*.pyc
```

---

# 📦 Requirements

Core dependencies include:

- Streamlit
- LangChain
- LangChain Community
- LangChain Text Splitters
- ChromaDB
- Groq
- Azure AI Translator
- PyPDF
- python-docx
- requests
- pydantic


# 🎯 How to Use

The application follows a simple and intuitive workflow.

---

## Step 1 — Upload a Document

Supported formats:

- PDF
- DOCX
- TXT

Click **Browse Files** and select your document.

The system automatically extracts text and prepares it for processing.

---

## Step 2 — Select Translation Language

Choose any supported Indian regional language.

Current supported languages include:

- Hindi
- Bengali
- Tamil
- Telugu
- Marathi
- Gujarati
- Kannada
- Malayalam
- Punjabi
- Odia
- Assamese
- Urdu
- Nepali
- Kashmiri
- Sindhi
- Konkani

---

## Step 3 — Start Processing

Click

```
🚀 Process Document
```

The application performs the following tasks automatically:

1. Extract text
2. Split text into chunks
3. Store embeddings in ChromaDB
4. Generate Summary
5. Generate Quiz
6. Translate Summary
7. Translate Quiz
8. Translate Original Document

---

# 📚 Processing Pipeline

```
Document Upload
        │
        ▼
Text Extraction
        │
        ▼
Text Chunking
        │
        ▼
Store in ChromaDB
        │
        ▼
Groq LLM
   │      │
   │      │
Summary   Quiz
   │      │
   └──┬───┘
      ▼
Azure AI Translator
      │
      ▼
Translated Output
```

---

# 🧠 AI Workflow

## Document Processing

The uploaded document is converted into plain text.

Supported loaders include:

- PyPDF
- python-docx

---

## Text Chunking

Large documents are divided into smaller chunks using

```
RecursiveCharacterTextSplitter
```

Benefits:

- Better retrieval
- Lower token usage
- Faster AI responses

---

## Vector Database

Each chunk is converted into embeddings and stored inside ChromaDB.

This enables semantic search during Question Answering.

---

## Summary Generation

Groq Llama 3.3 generates concise summaries while preserving important concepts.

The summary focuses on:

- Important topics
- Key definitions
- Core concepts
- Important facts

---

## Quiz Generation

The AI automatically creates

- Multiple Choice Questions

Each question contains

- Question
- Four options
- Correct answer
- Explanation

making the application useful for revision.

---

## Translation

Instead of running large local translation models,

the project uses

**Microsoft Azure AI Translator**

Benefits:

- Faster
- Lightweight
- Cloud-based
- Better deployment experience
- Lower RAM usage
- No GPU requirement

---

# 💬 Document Question Answering

After processing,

users can ask questions such as:

```
What is Machine Learning?
```

```
Explain the conclusion.
```

```
Summarize Chapter 5.
```

```
What are the key advantages?
```

The application retrieves relevant document chunks from ChromaDB before sending context to Groq.

This improves answer quality and reduces hallucinations.

---

# 📊 Features

| Feature | Supported |
|----------|-----------|
| PDF Upload | ✅ |
| DOCX Upload | ✅ |
| TXT Upload | ✅ |
| AI Summary | ✅ |
| AI Quiz | ✅ |
| AI Question Answering | ✅ |
| Azure Translation | ✅ |
| 16 Indian Languages | ✅ |
| ChromaDB Storage | ✅ |
| Docker Deployment | ✅ |
| AWS EC2 Deployment | ✅ |

---

# ⚡ Performance

Compared to the previous architecture using local translation models,

the current implementation offers:

- Lower memory usage
- Smaller Docker image
- Faster translation
- Better scalability
- Cloud-native deployment

---

# 📷 Suggested Screenshots

Add screenshots inside

```
assets/
```

Example:

```
assets/

├── home.png

├── upload.png

├── summary.png

├── quiz.png

├── translation.png

├── qa.png
```

Then include them in README.

Example:

```markdown
## Home Screen

![Home](assets/home.png)

---

## Summary Generation

![Summary](assets/summary.png)

---

## Quiz Generation

![Quiz](assets/quiz.png)

---

## Translation

![Translation](assets/translation.png)

---

## Question Answering

![Q&A](assets/qa.png)
```

---

# 🛠 Troubleshooting

## Azure Translation Not Working

Check

- Translator Key
- Translator Endpoint
- Translator Region

inside

```
.env
```

---

## Groq Error

Verify

```
GROQ_API_KEY
```

is valid.

---

## ChromaDB Error

Delete

```
chroma_db/
```

and restart the application.

A fresh database will be created automatically.

---

## Docker Issues

Rebuild the image:

```bash
docker build --no-cache -t ibm-study-bot .
```

---

## Port Already in Use

Run

```bash
streamlit run streamlit_study_bot.py --server.port 8502
```

or stop the existing process using port **8501**.

---

# 📈 Future Improvements

Some planned enhancements include:

- Voice-based Question Answering
- Text-to-Speech
- OCR for scanned documents
- Image-based document understanding
- PPT generation
- Flashcard generation
- Learning analytics dashboard
- User authentication
- Cloud database integration
- Mobile application

---

# 🤝 Contributing

Contributions are always welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature/your-feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/your-feature-name
```

5. Create a Pull Request

---

# 🌟 Why This Project?

Millions of students in India study in regional languages while educational resources are often available only in English.

This project aims to bridge that gap by using Artificial Intelligence to generate personalized study material and translate it into regional languages, making learning more accessible.

---

# 🚀 Future Roadmap

### ✅ Current Features

- AI-powered Summary Generation
- AI-powered Quiz Generation
- Retrieval-Augmented Document Q&A
- Azure AI Translation
- Docker Deployment
- AWS EC2 Compatibility
- ChromaDB Vector Search

---

## 🔜 Planned Features

### 🎙 Voice Assistant

- Voice-based Question Answering
- Speech-to-Text
- Text-to-Speech

---

### 📷 OCR Support

Support scanned PDFs using OCR.

Planned integrations:

- EasyOCR
- PaddleOCR

---

### 🖼 Image Understanding

Extract information directly from

- Charts
- Tables
- Diagrams
- Images

using Vision Language Models.

---

### 📚 Flashcard Generation

Generate revision flashcards automatically from uploaded documents.

---

### 📊 Learning Dashboard

Provide insights such as

- Time spent
- Quiz performance
- Weak topics
- Progress tracking

---

### 📱 Mobile Application

Future Android and iOS application.

---

### 👥 User Authentication

Support for

- Google Login
- Microsoft Login
- GitHub Login

---

### ☁ Cloud Database

Replace local storage with cloud databases such as

- MongoDB Atlas
- PostgreSQL
- Supabase

---

### 📈 Analytics

Track

- Popular topics
- Frequently asked questions
- User engagement
- Learning statistics

---

# 🏆 Project Highlights

✔ AI-Powered Educational Platform

✔ Retrieval-Augmented Generation (RAG)

✔ Cloud-based Translation using Azure AI Translator

✔ Multilingual Learning

✔ Dockerized Deployment

✔ AWS EC2 Ready

✔ Modular Architecture

✔ Scalable Design

✔ Modern Python Stack

✔ Industry-standard AI APIs

---

# 📋 Tech Summary

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Groq Llama 3.3-70B |
| Translation | Azure AI Translator |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Containerization | Docker |
| Cloud | AWS EC2 |
| Version Control | Git & GitHub |

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Cloud AI Services
- REST APIs
- Docker
- AWS EC2 Deployment
- Environment Variable Management
- Streamlit Development
- Production-ready AI Application Design

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use, modify and distribute it in accordance with the license terms.

---

# 🙏 Acknowledgements

This project makes use of several excellent open-source technologies and cloud services.

Special thanks to:

- Groq
- Microsoft Azure AI Translator
- LangChain
- ChromaDB
- Streamlit
- Python Community

---

# ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates further development.

---

# 📬 Contact

**Bhavya Batra**

B.Tech Artificial Intelligence & Machine Learning

University School of Automation & Robotics (USAR)

Guru Gobind Singh Indraprastha University

GitHub: https://github.com/<your-github>

LinkedIn: https://linkedin.com/in/<your-linkedin>

Email: your-email@example.com

---

# 📌 Citation

If you use this project in your research or academic work, please consider citing it.

```text

Regional Language Study Bot:
AI-powered multilingual document processing platform
using Groq, Azure AI Translator and LangChain.
2026.
```

---

## ⭐ Star History

If this project helped you, don't forget to leave a ⭐ on GitHub!

---

<p align="center">

Made with ❤️ using

**Python • Streamlit • Groq • Azure AI Translator • LangChain • ChromaDB • Docker • AWS**

</p>
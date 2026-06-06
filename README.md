# 🩺 Multimodal AI Medical Lab Analyzer

An AI-powered Medical Lab Report Analyzer that uses Multimodal Vision models (Google Gemini) and a local RAG (Retrieval-Augmented Generation) pipeline to extract complex blood test data and provide personalized, grounded dietary advice based on medical reference guidelines.

## 🌟 Features
- **Multimodal Ingestion**: Upload images (`.png`, `.jpg`) or `.pdf` lab reports.
- **Vision Extraction**: Uses Gemini Flash to read messy, unstructured medical tables and extract precise biomarkers (Test Name, Value, Unit, Status) into structured JSON.
- **Local RAG Pipeline**: Uses ChromaDB and HuggingFace Embeddings (`all-MiniLM-L6-v2`) to store and retrieve standard medical reference ranges and dietary guidelines.
- **Personalized Reasoning**: Automatically filters for abnormal (High/Low) results and generates actionable lifestyle and dietary suggestions based *strictly* on retrieved knowledge.
- **Streamlit UI**: A clean, intuitive web interface.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Orchestration**: LangChain (`langchain-core`, `langchain-chroma`)
- **Vision & Reasoning LLM**: Google Gemini (`gemini-flash-latest`) via `langchain-google-genai`
- **Vector Database**: ChromaDB (Runs locally)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Multimodal-AI-Assistant
   ```

2. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API Key:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

5. **Initialize the Vector Database:**
   Run the ingestion script to chunk and embed the medical reference guide into your local ChromaDB.
   ```bash
   python ingest.py
   ```

6. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## ⚠️ Disclaimer
This is an experimental AI application built for educational purposes. The dietary suggestions are based on general guidelines and should not replace professional medical advice. Always consult a physician or healthcare provider regarding any health-related questions.

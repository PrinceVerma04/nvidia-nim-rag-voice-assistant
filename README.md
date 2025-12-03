# 🤖 NVIDIA NIM RAG Voice Assistant
<img width="1926" height="881" alt="image" src="https://github.com/user-attachments/assets/b925aa43-0b02-4bf1-8ee0-c49823c72299" />

A Retrieval-Augmented Generation (RAG) application using NVIDIA NIM APIs. It supports PDF document QA, FAISS-based retrieval with NVIDIA embeddings, and has optional voice transcription (NVIDIA Canary ASR).

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-NIM-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com/)

## ✨ Features

- PDF document processing and chunking
- Context-aware Q&A over your documents
- Optional voice input/transcription via NVIDIA Canary ASR
- FAISS vector store for fast retrieval
- Conversational chat UI built with Streamlit
- Source attribution for each answer

## 🏗️ Architecture

User Query (Text/Voice) → Voice Transcription (Canary ASR) → Document Retrieval (FAISS + NVIDIA Embeddings) → Answer Generation (DeepSeek v3.1 via NVIDIA NIM) → Formatted Response + Sources

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA NIM API keys (LLM and Canary ASR)
- Windows / Linux / macOS

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/nvidia-nim-rag-voice-assistant.git
cd nvidia-nim-rag-voice-assistant
```

2. Create and activate a virtual environment

Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up API keys

Create `API_Key.txt` in the project root with:
```
LLM_API_KEY=nvapi-YOUR-LLM-API-KEY-HERE
CANARY_API_KEY=nvapi-YOUR-CANARY-API-KEY-HERE
```

Get your API keys:
- LLM API (DeepSeek): https://build.nvidia.com/deepseek-ai/deepseek-v3-1-terminus
- Canary ASR: https://build.nvidia.com/nvidia/canary-1b-asr

5. Add PDF documents
```bash
mkdir pdf
# copy your PDFs into the pdf/ folder
```

6. Run the app
```bash
streamlit run final_new_app.py
```
Open http://localhost:8501

## 📖 Usage

Text Q&A:
- Click "🔄 Initialize Vector Store"
- Wait for processing
- Ask questions via chat input
- Expand "View Source Documents" to see sources

Voice Q&A (if enabled):
- Use the Voice Chat tab (if added) to record or upload audio
- Transcribe using Canary ASR
- Submit transcribed text as the question

## 📁 Project Structure

nvidia-nim-rag-voice-assistant/
- final_new_app.py — main Streamlit application
- API_Key.txt — API keys (do not commit)
- requirements.txt — Python dependencies
- pdf/ — PDF documents
- README.md — this file
- .gitignore — common ignores
- LICENSE — project license

## 🛠️ Configuration

Edit `final_new_app.py` to change:
- LLM model parameters (model, temperature, top_p, max_completion_tokens)
- Embeddings model
- RAG chunk_size and chunk_overlap
- retrieval_k (number of docs to retrieve)

## 📦 Dependencies (examples)
See `requirements.txt` for pinned versions. Important entries:
- streamlit
- langchain-nvidia-ai-endpoints
- langchain-community
- langchain-core
- faiss-cpu
- pypdf
- nvidia-riva-client
- streamlit-mic-recorder
- protobuf

## 🚢 Deployment

Streamlit Cloud / Railway / Docker examples are supported. Provide LLM and CANARY API keys via environment variables or the `API_Key.txt` file when deploying.

## 🐛 Troubleshooting

- "No PDF documents found": ensure PDFs are in the `pdf/` folder
- "Invalid API key": verify keys in `API_Key.txt`, keys must start with `nvapi-`
- Protobuf issues: upgrade streamlit and protobuf (`pip install --upgrade streamlit "protobuf>=5.26.1"`)

## 🤝 Contributing

Fork → branch → commit → PR. Follow standard contribution workflow.

## 📄 License

MIT License — see LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA NIM for LLM and ASR
- LangChain for RAG primitives
- Streamlit for the UI

---
Made with ❤️ using NVIDIA NIM and Streamlit


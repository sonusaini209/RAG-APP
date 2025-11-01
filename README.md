# 📚 Smart Document QA Assistant

A **Retrieval-Augmented Generation (RAG)** Streamlit app that lets you upload any document (PDF or TXT) and ask questions about it using OpenAI GPT models.

---

## 🚀 Features
- Upload PDF or TXT files.
- Clean, concise AI answers.
- Built-in document text cleaning.
- Optional Debug Mode to inspect retrieved chunks.
- One-click deploy on Render.

---

## 🧩 Tech Stack
- **LangChain** – for the RAG pipeline
- **OpenAI GPT-4o-mini** – for LLM responses
- **FAISS** – for vector search
- **Streamlit** – for the UI
- **Render** – for deployment

---

## ⚙️ Local Setup

```bash
git clone https://github.com/<your-username>/smart-doc-qa.git
cd smart-doc-qa
pip install -r requirements.txt
cp .env.example .env

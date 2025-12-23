# 🤖 RAG PDF Chatbot – Retrieval-Augmented Generation

An AI-powered **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload PDF documents and ask natural language questions.  
The system retrieves the most relevant document context using **semantic search** and generates accurate, grounded answers using a **large language model (LLM)**.

---

## 📌 Features

- 📄 Upload and chat with PDF documents  
- 🧠 Semantic search using vector embeddings  
- 🔍 Context retrieval with cosine similarity (RAG pipeline)  
- 🤖 LLM-based answer generation via **Groq API**  
- 🎛️ Interactive web UI built with **Gradio**  
- 🎨 Optional custom CSS for clean and modern interface  

---

## 🧠 How It Works (RAG Pipeline)

1. **PDF Ingestion**  
   - Extracts text from uploaded PDF files

2. **Text Chunking & Embedding**  
   - Splits text into smaller chunks  
   - Converts chunks into vector embeddings using Sentence Transformers

3. **Retrieval**  
   - Computes cosine similarity between the user query and document embeddings  
   - Retrieves the most relevant chunks

4. **Generation**  
   - Sends retrieved context + user query to the LLM  
   - Generates a context-aware, accurate response

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|------------|
| Frontend UI | Gradio |
| LLM | Groq (LLaMA-based models) |
| Embeddings | Sentence-Transformers |
| Vector Search | Cosine Similarity (scikit-learn) |
| PDF Parsing | PyPDF |
| Language | Python |

---

## 📂 Project Structure

```bash
rag_chatbot/
│
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
└──README.md              # Project documentation


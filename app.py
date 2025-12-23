import os
import time
import gradio as gr
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# =====================================================
# LOAD MODELS & CLIENT
# =====================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found")

client = Groq(api_key=api_key)

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================
def extract_text_from_pdfs(files):
    all_text = ""
    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
        except Exception as e:
            return f"⚠️ Error reading PDF: {e}"
    return all_text.strip()

# =====================================================
# TEXT CHUNKING
# =====================================================
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# =====================================================
# EMBEDDINGS
# =====================================================
def embed_chunks(chunks):
    return embedder.encode(chunks)

# =====================================================
# RETRIEVAL
# =====================================================
def retrieve_relevant_chunks(question, chunks, embeddings, top_k=3):
    q_emb = embedder.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:]
    return [chunks[i] for i in top_indices]

# =====================================================
# GROQ COMPLETION
# =====================================================
def ask_groq(context, question):
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If not found, say:
"The answer is not available in the provided documents."

Context:
{context}

Question:
{question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content

# =====================================================
# RAG PIPELINE
# =====================================================
def rag_chatbot(files, question):
    if not files:
        return "⚠️ Please upload PDFs first."

    text = extract_text_from_pdfs(files)
    if not text:
        return "⚠️ No readable text found in PDFs."

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    relevant_chunks = retrieve_relevant_chunks(question, chunks, embeddings)

    return ask_groq("\n\n".join(relevant_chunks), question)

# =====================================================
# STREAMING CHAT HANDLER (MESSAGE FORMAT)
# =====================================================
def chat_handler_stream(message, history, files):
    if history is None:
        history = []

    # user message
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "💭 Thinking..."})
    yield history

    answer = rag_chatbot(files, message)

    streamed = ""
    for ch in answer:
        streamed += ch
        history[-1]["content"] = streamed
        time.sleep(0.01)
        yield history

# =====================================================
# UI
# =====================================================
with gr.Blocks() as app:

    gr.Markdown(
        "<h2 style='text-align:center;'>📄 Document-Grounded Chat Assistant (Groq)</h2>"
    )

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload PDFs")
            pdf_files = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520)
            user_input = gr.Textbox(
                placeholder="Ask something from your documents…",
                show_label=False
            )
            send_btn = gr.Button("Send")

    send_btn.click(
        chat_handler_stream,
        inputs=[user_input, chatbot, pdf_files],
        outputs=chatbot
    )

    user_input.submit(
        chat_handler_stream,
        inputs=[user_input, chatbot, pdf_files],
        outputs=chatbot
    )

app.launch()

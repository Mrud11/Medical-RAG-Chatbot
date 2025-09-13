"""
Minimal Streamlit app that loads FAISS index + metadata and answers queries using OpenAI / Groq.
Put your API key in environment variables or Streamlit secrets.
"""
import os, pickle
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ---- Colab watcher fix ----
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_DISABLE_FILE_WATCHER", "true")

# ---- UI ----
st.set_page_config(page_title="🩺 Medical RAG Chatbot")
st.title("🩺 Medical RAG Chatbot")
st.write("Type your medical question below:")

# ---- Load index & meta (ensure path correct) ----
INDEX_PATH = "data/medquad_faiss.index"
META_PATH = "data/medquad_meta.pkl"

@st.cache_resource
def load_index_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

index, meta = load_index_meta()

# ---- Embedding model (local) ----
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

def normalize(vects):
    norms = np.linalg.norm(vects, axis=1, keepdims=True)
    norms[norms==0]=1e-10
    return vects / norms

def retrieve(query, top_k=4):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = normalize(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        row = meta.iloc[idx]
        results.append({"text": row['text'], "score": float(score)})
    return results

# ---- LLM call (OpenAI example) ----
import openai
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def build_context_text(retrieved):
    pieces = []
    for i, r in enumerate(retrieved):
        pieces.append(f"[Source {i+1}] {r['text']}")
    return "\n\n".join(pieces)

def generate_answer_openai(user_question, retrieved):
    if not OPENAI_KEY:
        return "No OPENAI_API_KEY provided. Set it as env var or use Streamlit secrets."
    context_text = build_context_text(retrieved)
    system_msg = (
        "You are an assistant that answers medical questions using ONLY the provided context. "
        "If missing info, advise consulting a medical professional."
    )
    user_prompt = f"Question: {user_question}\n\nContext:\n{context_text}"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_prompt}],
        max_tokens=300, temperature=0.0
    )
    return resp['choices'][0]['message']['content'].strip()

# ---- Chat UI logic ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "You" if msg["role"]=="user" else "Assistant"
    st.markdown(f"**{role}:** {msg['content']}")

user_input = st.chat_input("Ask a medical question:")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    retrieved = retrieve(user_input, top_k=4)
    if len(retrieved)==0:
        answer = "No relevant context found."
    else:
        # call LLM
        try:
            answer = generate_answer_openai(user_input, retrieved)
        except Exception as e:
            answer = f"[LLM error] {e}"
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.experimental_rerun()


# Medical-RAG-Chatbot
Medical RAG Chatbot using MedQuAD dataset, FAISS retrieval, and LLM (OpenAI/Groq) for contextual AI-powered medical question answering

A Retrieval-Augmented Generation (RAG) chatbot for medical FAQs using the MedQuAD dataset.  
User types a medical question → app retrieves relevant context from the knowledge base (FAISS) → passes context to an LLM (OpenAI / Groq) → returns a contextual, concise answer.

---

## Deliverables in this repository
- `src/build_index.py` — preprocess dataset, chunk text, produce embeddings and FAISS index.
- `src/app.py` — Streamlit UI + retrieval + LLM generation.
- `data/medquad.csv` — (not included) Place dataset here or mount Google Drive.
- `requirements.txt` — Python dependencies.
- `README.md` — setup & run instructions (this file).
- Short design decisions & troubleshooting guidance.

---

## Quick demo (what you should see)
1. Open the Streamlit UI (runs on port `8501`).
2. Type: `What are the early symptoms of diabetes?`
3. The app retrieves top-k documents from FAISS, sends them to the LLM, and shows a concise answer citing the source indices.

---

## Setup — Local (Linux / macOS / WSL)
```bash
# 1. clone
git clone https://https://github.com/Mrud11/Medical-RAG-Chatbot.git
cd medical-rag-chatbot

# 2. python env
python -m venv .venv
source .venv/bin/activate

# 3. install
pip install -r requirements.txt

# 4. place dataset
# Put medquad.csv into data/ or update path in src/build_index.py

# 5. build index (generates faiss index + metadata)
python src/build_index.py --data_path data/medquad.csv --out_dir ./data

# 6. run app
streamlit run src/app.pyz





# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for medical FAQs using the MedQuAD dataset.  
User types a medical question → app retrieves relevant context from the knowledge base (FAISS) → passes context to an LLM (OpenAI) → returns a contextual, concise answer.

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
## Repository Structure

# Medical RAG Chatbot — RAG (FAISS) + LLM (OpenAI) for medical FAQs.
Repo layout:
medical-rag-chatbot/
├─ data/medquad.csv
├─ src/
│  ├─ build_index.py
│  └─ app.py
├─ requirements.txt
├─ README.md
└─ LICENSE (MIT)



## Setup — Local (Linux / macOS / WSL)
```bash
# 1. clone
git clone https://github.com/<you>/medical-rag-chatbot.git
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
streamlit run src/app.py

```
---
# 7. How it’s built — high level pipeline

- Load dataset (CSV) → ensure columns question, answer or auto-detect first two columns.
- Preprocess & chunk: split QA text into manageable chunks (sentence-based chunks ~150–200 words).
- Embeddings: compute vector embeddings for chunks (sentence-transformers or OpenAI embeddings).
- Vector index: normalize embeddings (for cosine) and save with FAISS (IndexFlatIP or IndexFlatL2).
- Query-time:
- Embed user query.
- Retrieve top-k chunks from FAISS.
- Build a short “context” string (source tags + text).
- Call LLM with system prompt that instructs: “use ONLY this context; do not hallucinate; recommend consulting a doctor when needed.”
- UI: present answer and show retrieved sources (optional expander) and include medical disclaimer.

# 8. Safety & ethics (must have)

- Always show a prominent disclaimer: not medical advice; consult professionals.

- Use low sampling (temperature=0.0–0.2) to reduce hallucinations.

- Log queries & responses for audit and quality review (anonymize PII).

- For production clinical use, require human oversight, clinical validation, and regulatory compliance (HIPAA/GDPR etc.).

"""
build_index.py
- Loads CSV (auto-detects first 2 columns as question/answer if names differ)
- Chunks text
- Computes embeddings (option: sentence-transformers or OpenAI embeddings)
- Builds FAISS index and saves index + metadata
"""
import argparse, os, pickle
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import nltk
nltk.download('punkt')

def load_csv(path):
    df = pd.read_csv(path)
    # Ensure at least 2 columns; take first two if not named Question/Answer
    if "question" in df.columns.str.lower() or "Question" in df.columns:
        # normalize column names if present
        qcol = [c for c in df.columns if c.lower().startswith("question")][0]
        acol = [c for c in df.columns if c.lower().startswith("answer")][0]
        df = df[[qcol, acol]]
        df.columns = ["question", "answer"]
    else:
        df = df.iloc[:, :2]
        df.columns = ["question", "answer"]
    df = df.dropna().reset_index(drop=True)
    return df

def chunk_text(text, max_words=200):
    sents = sent_tokenize(text)
    chunks = []
    cur = ""
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur_words + w <= max_words:
            cur += " " + s
            cur_words += w
        else:
            chunks.append(cur.strip())
            cur = s
            cur_words = w
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def build(args):
    df = load_csv(args.data_path)
    # create chunks
    rows = []
    for i, row in df.iterrows():
        text = f"Q: {row['question']}\nA: {row['answer']}"
        chunks = chunk_text(text, max_words=200)
        for j, c in enumerate(chunks):
            rows.append({"doc_id": i, "chunk_id": f"{i}_{j}", "text": c, "question": row['question']})
    meta = pd.DataFrame(rows)
    print("Chunks:", len(meta))

    # embeddings (sentence-transformers)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = meta['text'].tolist()
    embs = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine (optional)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0] = 1e-10
    embs = embs / norms

    # build FAISS
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    os.makedirs(args.out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out_dir, "medquad_faiss.index"))
    with open(os.path.join(args.out_dir, "medquad_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Saved index and metadata.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()
    build(args)

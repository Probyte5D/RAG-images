import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# Modello embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index.index"
DOCS_PATH = "documents.pkl"

def load_or_create_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)  # 384 = dimensione all-MiniLM
        documents = []
    return index, documents

def save_index(index, documents):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

def add_text(text):
    index, documents = load_or_create_index()
    embedding = embedder.encode([text])
    index.add(embedding)
    documents.append(text)
    save_index(index, documents)

def search_similar(query, top_k=3):
    index, documents = load_or_create_index()
    embedding = embedder.encode([query])
    D, I = index.search(embedding, top_k)
    return [documents[i] for i in I[0] if i < len(documents)]

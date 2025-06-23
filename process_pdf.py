import os
import fitz  # PyMuPDF
from uuid import uuid4
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Initialize ChromaDB ===
client = PersistentClient(path="./db")
collection = client.get_or_create_collection("budget")

# === Function: Cleanly chunk large text into word-limited passages ===
def chunk_text(text, max_words=200):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# === Function: Extract text chunks from a single PDF ===
def load_budget_chunks(pdf_path):
    chunks = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            for chunk in chunk_text(text, max_words=200):
                if len(chunk.strip()) > 50:
                    chunks.append((chunk.strip(), page_num + 1))
    return chunks

# === Function: Index a single PDF file into ChromaDB ===
def index_budget(pdf_path):
    print(f"📄 Processing: {pdf_path}")
    chunks = load_budget_chunks(pdf_path)
    texts = [c[0] for c in chunks]
    pages = [c[1] for c in chunks]
    embeddings = model.encode(texts).tolist()

    for i, text in enumerate(texts):
        doc_id = f"{os.path.basename(pdf_path)}_{i}"
        metadata = {
            "source": os.path.basename(pdf_path),
            "page": pages[i]
        }
        collection.add(documents=[text], embeddings=[embeddings[i]], ids=[doc_id], metadatas=[metadata])
    
    print(f"✅ Indexed {len(chunks)} chunks from {pdf_path}.")

# === MAIN ===
if __name__ == "__main__":
    # List all budget PDFs you want to include
    pdf_files = [
        "budget_brief_25_26.pdf",
        "annual_budget_25_26.pdf",
        "receipts_memorandum_25_26.pdf",
        "finance_bill_25_26.pdf"
    ]

    for pdf in pdf_files:
        if os.path.exists(pdf):
            index_budget(pdf)
        else:
            print(f"❌ File not found: {pdf}")

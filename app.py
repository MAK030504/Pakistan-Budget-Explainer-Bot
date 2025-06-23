import os
import streamlit as st
import requests
import re
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# === Load environment variables ===
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

# === Load embedder ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = embedder.to("cpu")

# === Connect to ChromaDB ===
client = PersistentClient(path="./db")
collection = client.get_or_create_collection("budget")

# === Extract keywords from query ===
def extract_concepts(query):
    stopwords = {"what", "how", "is", "are", "the", "a", "on", "of", "in", "for", "to", "does", "do", "it", "there", "any"}
    words = re.findall(r"\b\w+\b", query.lower())
    return [[word] for word in words if word not in stopwords and len(word) > 2]

# === Filter documents using concept groups ===
def filter_advanced(query, docs, required_concepts, min_matches=3):
    filtered = []
    for doc in docs:
        matches = sum(any(word in doc.lower() for word in group) for group in required_concepts)
        if matches >= min_matches:
            filtered.append(doc)
    return filtered

# === Together API summarization ===
def summarize_with_together(query, excerpts):
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
    f"You are a helpful assistant for explaining Pakistan's 2025–26 federal budget. "
    f"Based on the excerpts below(use your own info to bring coherence where needed), answer the user's question as clearly and informatively as possible.\n\n"
    
    f"Question: {query}\n\n"
    f"Excerpts:\n{excerpts}\n\n"
    
    f"---\n"
    f"Guidelines:\n"
    f"- Provide a short and **specific** answer.\n"
    f"- If the answer is found in the excerpts, include details such as tax rates, exemptions, or amounts.\n"
    f"- If the answer is **not** clearly found, you may infer based on the text — but indicate uncertainty (e.g., 'likely', 'possibly').\n"
    f"- Avoid repeating the excerpts verbatim unless needed for clarity.\n"
    )

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0,
        "top_p": 0.85
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['text'].strip()
    except requests.exceptions.RequestException as e:
        return f"❌ API request failed: {str(e)}"
    except KeyError:
        return "❌ Unexpected response format from Together API."

# === Streamlit UI ===
st.set_page_config(page_title="Pakistan Budget Explainer", page_icon="📊")
st.title("📊 Pakistan Budget Explainer (2025–26)")

st.markdown("""### 📌 FY 2025–26 At a Glance
- **Total Budget Outlay:** ≈ ₨ 17.6 trillion  
- **GDP Growth Target:** 4.2%  
- **Defence Spending:** +20% (≈ ₨ 2.55 trillion)  
- **Fiscal Deficit Target:** ~3.9% of GDP
""")

st.markdown("Ask anything about the Pakistan federal budget. For example:\n"
            "- *What subsidies are available for farmers?*\n"
            "- *How much is allocated for higher education?*\n"
            "- *Is there a tax on solar panels?*")

user_query = st.text_input("🔍 Your Question:")

if st.button("Search and Explain"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔎 Searching the budget documents..."):
            embedded_query = embedder.encode([user_query])[0]
            results = collection.query(query_embeddings=[embedded_query], n_results=8)

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        if not documents:
            st.info("❌ No relevant information found. Try rephrasing your question.")
        else:
            required_concepts = extract_concepts(user_query)
            filtered_docs = filter_advanced(user_query, documents, required_concepts, min_matches=2)
            if not filtered_docs:
                filtered_docs = documents

            st.markdown("### 📄 Relevant Budget Excerpts:")
            for i, doc in enumerate(filtered_docs[:5]):
                meta = metadatas[i] if i < len(metadatas) else {}
                source = meta.get("source", "Unknown")
                page = meta.get("page", "?")
                st.markdown(f"**Source:** `{source}` – Page `{page}`")
                st.code(doc[:1000] + ("..." if len(doc) > 1000 else ""), language="markdown")

            combined_passages = "\n\n".join(filtered_docs[:5])

            with st.spinner("🧠 Simplifying for you..."):
                simplified = summarize_with_together(user_query, combined_passages)

            st.markdown("### 💡 Simplified Explanation:")
            st.write(simplified)

st.markdown("---")
st.caption("Built by Mustafa Avais using ChromaDB, SentenceTransformers, and Together.ai 🚀")

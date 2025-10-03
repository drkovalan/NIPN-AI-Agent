import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from io import BytesIO
from openrouter import OpenRouterClient

# -------------------------------
# 1. Setup OpenRouter API Key
# -------------------------------
API_KEY = st.secrets.get("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Missing OpenRouter API Key. Add it in Streamlit Secrets.")
    st.stop()

client = OpenRouterClient(api_key=API_KEY)

# -------------------------------
# 2. Initialize Chroma DB
# -------------------------------
chroma_client = chromadb.Client()
embeddings_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="mistralai/mistral-7b-instruct:free"
)

# Create or get collection
collection_name = "nipn_docs"
try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embeddings_fn
    )

# -------------------------------
# 3. Streamlit UI for uploads
# -------------------------------
st.title("NIPN AI Knowledge Agent")
st.markdown("Upload PDFs, Excel files, or provide URLs for NIPN knowledge base")

uploaded_files = st.file_uploader("Upload PDF/Excel", type=["pdf", "xlsx"], accept_multiple_files=True)
url_input = st.text_area("Or paste URLs (one per line)")

documents = []
metadata = []

# -------------------------------
# 4. Process uploaded files
# -------------------------------
if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            documents.append(text)
            metadata.append({"source": file.name})
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(file)
            text = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
            documents.append(text)
            metadata.append({"source": file.name})

# -------------------------------
# 5. Process URLs
# -------------------------------
if url_input:
    for url in url_input.split("\n"):
        url = url.strip()
        if url:
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(separator="\n")
                documents.append(text)
                metadata.append({"source": url})
            except:
                st.warning(f"Failed to load URL: {url}")

# -------------------------------
# 6. Add documents to Chroma
# -------------------------------
if documents:
    collection.add(documents=documents, metadatas=metadata, ids=[str(i) for i in range(len(documents))])
    st.success(f"{len(documents)} documents added to Chroma knowledge base!")

# -------------------------------
# 7. Query AI
# -------------------------------
query = st.text_input("Ask NIPN AI about nutrition data, reports, or findings:")

if query and documents:
    # Perform vector search
    results = collection.query(query_texts=[query], n_results=3)
    context_text = "\n".join([doc for doc in results['documents'][0]])

    prompt = f"""
    You are a Nutrition Data AI Assistant for NIPN Laos.
    Use the context below to answer user query professionally and concisely.

    Context:
    {context_text}

    User Question: {query}
    AI Answer:
    """

    response = client.chat.create(
        model="mistralai/mistral-7b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    answer = response.choices[0].message["content"]
    st.markdown("**Answer:**")
    st.write(answer)


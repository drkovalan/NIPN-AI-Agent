import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from openrouter import OpenRouterClient
import os
import pandas as pd

# -------------------------------
# 1. Set up API Key
# -------------------------------
API_KEY = st.secrets.get("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Missing OpenRouter API Key. Please add it in Streamlit Secrets.")
    st.stop()

client = OpenRouterClient(api_key=API_KEY)

# -------------------------------
# 2. Upload documents
# -------------------------------
st.title("NIPN AI Knowledge Agent")
st.markdown("Upload PDFs or Excel files for NIPN knowledge base")

uploaded_files = st.file_uploader("Upload PDF/Excel", type=["pdf", "xlsx"], accept_multiple_files=True)

documents = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            loader = PyPDFLoader(file)
            docs = loader.load()
            documents.extend(docs)
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(file)
            text = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
            documents.append(TextLoader(text).load()[0])

# -------------------------------
# 3. Split documents into chunks
# -------------------------------
if documents:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs_chunks = splitter.split_documents(documents)

    st.success(f"{len(docs_chunks)} chunks created from uploaded documents!")

    # -------------------------------
    # 4. Create FAISS vector store
    # -------------------------------
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vector_store = FAISS.from_documents(docs_chunks, embeddings)
    st.success("FAISS vector store created!")

# -------------------------------
# 5. Query AI
# -------------------------------
query = st.text_input("Ask NIPN AI about nutrition data, reports, or findings:")

if query and documents:
    # Search similar chunks
    docs_found = vector_store.similarity_search(query, k=3)
    context_text = "\n".join([d.page_content for d in docs_found])

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

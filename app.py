import streamlit as st
import os
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader, WebBaseLoader
from openrouter import OpenRouterClient

# ---- Streamlit Secrets ----
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# ---- LLM Setup (DeepSeek-V3 via OpenRouter) ----
client = OpenRouterClient(api_key=API_KEY)
MODEL_NAME = "deepseek/deepseek-chat-v3.1:free"

# ---- Embeddings Setup ----
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

st.title("📊 NIPN Knowledge AI Agent")

# Temporary folder for uploads
os.makedirs("temp", exist_ok=True)

# ---- Functions ----
def process_upload(file):
    path = os.path.join("temp", file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif file.name.endswith(".txt"):
        loader = TextLoader(path)
    elif file.name.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(path)
    else:
        st.error("Unsupported file type")
        return []
    return loader.load()

def fetch_wp_content(site_url):
    posts = []
    page = 1
    while True:
        url = f"{site_url}/wp-json/wp/v2/posts?per_page=100&page={page}"
        r = requests.get(url)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        for post in data:
            title = post.get('title', {}).get('rendered', '')
            content = post.get('content', {}).get('rendered', '')
            posts.append(title + "\n" + content)
        page += 1
    return posts

# ---- Inputs ----
uploaded_file = st.file_uploader("Upload PDF, Excel, or TXT", type=["pdf", "txt", "xlsx"])
wp_site = st.text_input("WordPress Site URL (e.g. https://nipn.lsb.gov.la)")
url_input = st.text_input("Or enter a webpage URL:")

# ---- Update Knowledge Base ----
if st.button("🔄 Update Knowledge Base"):
    all_docs = []

    # Uploaded file
    if uploaded_file is not None:
        all_docs += process_upload(uploaded_file)
        st.success(f"✅ {uploaded_file.name} processed!")

    # WordPress site
    if wp_site:
        wp_texts = fetch_wp_content(wp_site)
        all_docs += [{"page_content": t} for t in wp_texts]
        st.success(f"✅ Fetched {len(wp_texts)} posts from WordPress")

    # Single URL
    if url_input:
        loader = WebBaseLoader(url_input)
        all_docs += loader.load()
        st.success("✅ URL content added!")

    if all_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("faiss_index")
        st.success("✅ Knowledge base updated!")
    else:
        st.warning("No content to add. Please provide files, WordPress URL, or webpage.")

# ---- Query AI ----
query = st.text_input("Ask a question about NIPN knowledge base:")

if query:
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        results = db.similarity_search(query, k=3)
        context = "\n\n".join([r.page_content for r in results])

        # DeepSeek-V3 response
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a Nutrition Data AI Assistant for NIPN."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ],
        )
        st.write("### 🤖 AI Answer")
        st.write(response['choices'][0]['message']['content'])
    else:
        st.warning("Please update the knowledge base first.")


import streamlit as st
import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="NIPN AI Agent", layout="wide")

st.title("📊 NIPN AI Agent")
st.write("Ask questions, summarize reports, upload documents, or analyze data.")

# -------------------------------
# User Inputs
# -------------------------------
user_input = st.text_area("💬 Enter your query:", placeholder="Type your question here...")
uploaded_file = st.file_uploader("📂 Upload a document (PDF, TXT, CSV, Excel)", type=["pdf","txt","csv","xlsx"])

# -------------------------------
# OpenRouter Setup
# -------------------------------
API_KEY = os.environ.get("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")  # Set in Streamlit Secrets
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat-v3.1:free"  # You can change model here

def query_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {response.text}"

# -------------------------------
# Handle File Upload
# -------------------------------
file_text = ""
if uploaded_file:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        file_text = "[PDF Uploaded: content parsing not enabled here]"  # For advanced: use PyPDF2
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.subheader("📈 Uploaded CSV Preview")
        st.dataframe(df.head())

        # Auto-generate chart
        st.subheader("📊 Auto Chart")
        fig, ax = plt.subplots()
        df.select_dtypes(include=["number"]).plot(ax=ax)
        st.pyplot(fig)

        file_text = f"[CSV Uploaded with {df.shape[0]} rows and {df.shape[1]} columns]"
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        st.subheader("📈 Uploaded Excel Preview")
        st.dataframe(df.head())

        # Auto-generate chart
        st.subheader("📊 Auto Chart")
        fig, ax = plt.subplots()
        df.select_dtypes(include=["number"]).plot(ax=ax)
        st.pyplot(fig)

        file_text = f"[Excel Uploaded with {df.shape[0]} rows and {df.shape[1]} columns]"

# -------------------------------
# Submit Button
# -------------------------------
if st.button("🚀 Ask AI"):
    if not API_KEY:
        st.error("⚠️ Missing OpenRouter API Key. Please add it in Streamlit Secrets.")
    elif not user_input and not file_text:
        st.warning("Please enter a query or upload a file.")
    else:
        query = user_input + "\n\n" + file_text
        answer = query_openrouter(query)
        st.success(answer)


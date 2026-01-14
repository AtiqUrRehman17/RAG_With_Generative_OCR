import streamlit as st
import tempfile
import os

# import your existing class
from main import ScannedPDFChatbot

st.set_page_config(page_title="Scanned PDF RAG Chatbot", layout="wide")

st.title("📄 Scanned PDF RAG Chatbot")
st.write("Upload a scanned PDF and ask questions strictly from the document.")

# ---------- SESSION STATE ----------
if "bot" not in st.session_state:
    st.session_state.bot = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ---------- PDF UPLOAD ----------
uploaded_file = st.file_uploader("Upload a scanned PDF", type=["pdf"])

if uploaded_file and not st.session_state.initialized:
    with st.spinner("Saving PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

    st.success("PDF uploaded successfully.")

    if st.button("🚀 Initialize Chatbot"):
        with st.spinner("Running OCR, embedding, and indexing (this may take time)..."):
            bot = ScannedPDFChatbot(pdf_path)
            bot.initialize()

            st.session_state.bot = bot
            st.session_state.initialized = True

        st.success("Chatbot is ready!")

# ---------- CHAT INTERFACE ----------
if st.session_state.initialized:
    st.divider()
    st.subheader("💬 Ask Questions")

    user_question = st.text_input("Enter your question")

    if user_question:
        with st.spinner("Thinking..."):
            answer = st.session_state.bot.ask(user_question)

        st.markdown("### 🧠 Answer")
        st.write(answer)

# ---------- FOOTER ----------
st.divider()
st.caption("Answers are generated strictly from the uploaded scanned document.")

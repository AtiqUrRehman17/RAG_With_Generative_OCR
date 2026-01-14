import os
import hashlib
from langchain_openai import ChatOpenAI

from chatbot.ocr import load_and_ocr_pdf
from chatbot.chunking import split_documents
from chatbot.vectorstore import create_vectorstore
from chatbot.rag import setup_rag_chain


class ScannedPDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_id = hashlib.md5(pdf_path.encode()).hexdigest()

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

        self.ocr_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def initialize(self):
        docs = load_and_ocr_pdf(self)
        chunks = split_documents(docs)
        create_vectorstore(self, chunks)
        setup_rag_chain(self)
        print("\n✅ Chatbot initialized successfully.")

    def ask(self, question):
        return self.rag_chain.invoke(question)

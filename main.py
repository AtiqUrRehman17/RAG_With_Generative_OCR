import os
import hashlib
import base64
import io
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


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

    # ---------- IMAGE UTILS ---------- #

    def image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ---------- OCR ---------- #

    def generative_ocr(self, image, page_number):
        image_b64 = self.image_to_base64(image)

        ocr_prompt = (
            "Extract ONLY the text that is CLEARLY visible in this image.\n\n"
            "Rules:\n"
            "- DO NOT guess missing text\n"
            "- DO NOT infer names, dates, or addresses\n"
            "- If text is unreadable, write: [UNCLEAR]\n"
            "- If a field is empty, write: [BLANK]\n"
            "- Preserve original wording exactly\n"
        )

        response = self.ocr_llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ]
        )

        text = response.content.strip()
        unclear_count = text.count("[UNCLEAR]")
        confidence = "low" if unclear_count > 8 else "high"

        return Document(
            page_content=text,
            metadata={
                "page": page_number,
                "pdf_id": self.pdf_id,
                "source_file": os.path.basename(self.pdf_path),
                "ocr_confidence": confidence,
            },
        )

    def load_and_ocr_pdf(self):
        print("Converting PDF to images...")
        images = convert_from_path(self.pdf_path, dpi=300)

        documents = []
        for i, image in enumerate(images, start=1):
            print(f"Running Generative OCR on page {i}")
            documents.append(self.generative_ocr(image, i))

        return documents

    # ---------- CHUNKING ---------- #

    def split_documents(self, documents):
        print("Splitting OCR text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    # ---------- VECTOR STORE ---------- #

    def create_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )

        self.vectorstore.add_documents(chunks)

        # ✅ FIXED FILTER SYNTAX
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 12,
                "lambda_mult": 0.6,
                "filter": {
                    "$and": [
                        {"pdf_id": self.pdf_id},
                        {"ocr_confidence": "high"},
                    ]
                },
            },
        )

        print("Vector store ready (Chroma filter fixed).")

    # ---------- RAG ---------- #

    def retrieve_context(self, question):
        docs = self.retriever.invoke(question)
        return "\n\n".join(
            f"[Page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
        )

    def setup_rag_chain(self):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are answering questions strictly from a scanned document.

Context:
{context}

Question:
{question}

Rules:
- Use ONLY the provided context
- Do NOT guess or infer
- If information is missing or unclear, say:
  "The document does not clearly state this information."

Answer:
"""
        )

        self.rag_chain = (
            {
                "context": lambda q: self.retrieve_context(q),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    # ---------- INIT ---------- #

    def initialize(self):
        docs = self.load_and_ocr_pdf()
        chunks = self.split_documents(docs)
        self.create_vectorstore(chunks)
        self.setup_rag_chain()
        print("\n✅ Chatbot initialized successfully.")

    # ---------- ASK ---------- #

    def ask(self, question):
        return self.rag_chain.invoke(question)


# ---------- CLI ---------- #

def main():
    print("=== Scanned PDF RAG Chatbot (STABLE BUILD) ===\n")

    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ File not found.")
        return

    bot = ScannedPDFChatbot(pdf_path)
    bot.initialize()

    print("\nType 'exit' to quit\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print(f"\nAssistant: {bot.ask(q)}\n")


if __name__ == "__main__":
    main()

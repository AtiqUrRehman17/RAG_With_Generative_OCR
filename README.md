# 📄 Scanned PDF RAG Chatbot (Generative OCR + Pinecone)

A **Retrieval-Augmented Generation (RAG)** based chatbot that allows you to **ask questions from scanned PDF documents** (image-based PDFs) using **Generative OCR**, **vector search**, and **LLMs**.

This project is designed for PDFs where traditional text extraction fails — such as **scanned contracts, invoices, reports, forms, or handwritten documents**.

---

## 🚀 Key Features

- ✅ **Generative OCR using GPT-4o-mini**
- ✅ Handles **scanned / image-only PDFs**
- ✅ **High-accuracy text extraction** with confidence filtering
- ✅ **Vector search using Pinecone**
- ✅ **RAG-based question answering**
- ✅ Strictly answers **only from document content**
- ✅ CLI-based interface (easy to extend to Streamlit / FastAPI)
- ✅ Clean, modular, production-ready architecture

---

## 🧠 How the Project Works (End-to-End)

### 1️⃣ PDF → Images
- The PDF is converted into high-resolution images using `pdf2image`.
- Each page is processed independently.

### 2️⃣ Generative OCR
- Each page image is sent to **GPT-4o-mini** as a vision input.
- The model extracts **only clearly visible text**.
- Strict rules are applied:
  - No guessing
  - No inference
  - `[UNCLEAR]` for unreadable text
  - `[BLANK]` for empty fields

### 3️⃣ OCR Confidence Scoring
- Pages with excessive `[UNCLEAR]` markers are tagged as **low confidence**.
- Only **high-confidence OCR chunks** are used for retrieval.

### 4️⃣ Text Chunking
- OCR text is split into overlapping chunks using:
  - RecursiveCharacterTextSplitter
  - Optimized chunk size and overlap for semantic retrieval

### 5️⃣ Vector Embeddings & Storage
- Text chunks are embedded using **OpenAI Embeddings**
- Stored in **Pinecone vector database**
- Each PDF gets its **own namespace**

### 6️⃣ Retrieval-Augmented Generation (RAG)
- User questions retrieve relevant chunks via **MMR search**
- Retrieved context is injected into a strict prompt
- The LLM:
  - Uses ONLY retrieved content
  - Refuses to guess if information is missing

---

## 🗂️ Project Folder Structure

```text
scanned_pdf_rag/
│
├── main.py
│
├── chatbot/
│   ├── __init__.py
│   ├── core.py          # Main chatbot orchestration
│   ├── ocr.py           # Generative OCR logic
│   ├── chunking.py      # Text splitting
│   ├── vectorstore.py   # Pinecone integration
│   └── rag.py           # RAG pipeline
│
├── utils/
│   ├── __init__.py
│   └── image_utils.py   # Image → Base64 helpers
│
└── .env


### 🛠️ Tech Stack
Core Technologies

Python 3.9+

LangChain

OpenAI GPT-4o-mini

Pinecone Vector Database

### Supporting Libraries

pdf2image

Pillow (PIL)

python-dotenv

hashlib

base64

🔑 Environment Variables

### Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key

### 📦 Installation
### 1️⃣ Clone the Repository
git clone https://github.com/your-username/scanned-pdf-rag-chatbot.git
cd scanned-pdf-rag-chatbot

### 2️⃣ Create Virtual Environment
python -m venv myenv
source myenv/bin/activate   # Linux / Mac
myenv\Scripts\activate      # Windows

### 3️⃣ Install Dependencies
pip install -r requirements.txt


Make sure Poppler is installed for pdf2image:

Windows: Add Poppler to PATH

Linux: sudo apt install poppler-utils

Mac: brew install poppler

### ▶️ How to Run
python main.py


You will be prompted to:

Enter the path to a scanned PDF

Ask questions interactively

### Example:

Enter the path to your PDF file: contract.pdf
You: What is the contract start date?
Assistant: The document does not clearly state this information.

### 🧪 Example Use Cases

📑 Legal contracts

🧾 Invoices & receipts

🏦 Bank statements

📜 Historical documents

🏥 Medical reports

📊 Scanned reports & forms

### 🔒 Safety & Accuracy Guarantees

❌ No hallucinations

❌ No guessing

❌ No external knowledge

✅ Answers strictly from document context

✅ Explicit response when data is missing or unclear

🧩 Extensibility

### This architecture is designed to be easily extended with:

🔹 Streamlit UI

🔹 FastAPI backend

🔹 Multi-PDF support

🔹 Persistent chat history

🔹 Metadata-based filtering

🔹 User authentication

### 📈 Why This Project Matters

Traditional OCR often fails on:

Poor scans

Complex layouts

Handwritten or noisy documents

This project solves that by combining:

Vision-capable LLMs

Vector databases

Strict RAG constraints

Result: Reliable, explainable, document-grounded AI answers.

### 🤝 Contributions

Contributions, issues, and feature requests are welcome.


### 🙌 Author

Atiq Ur Rehman
AI Engineer | RAG | OCR | LLM Systems

If you found this useful, feel free to ⭐ the repository and share it!

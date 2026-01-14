import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def create_vectorstore(bot, chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "ocr-rag"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    bot.vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=bot.pdf_id,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    bot.retriever = bot.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 12,
            "lambda_mult": 0.6,
            "filter": {"ocr_confidence": "high"},
        },
    )

    print("Vector store ready (Pinecone).")

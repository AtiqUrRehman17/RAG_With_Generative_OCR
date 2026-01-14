import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def retrieve_context(bot, question):
    docs = bot.retriever.invoke(question)
    return "\n\n".join(
        f"[Page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )


def setup_rag_chain(bot):
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

    bot.rag_chain = (
        {
            "context": lambda q: retrieve_context(bot, q),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

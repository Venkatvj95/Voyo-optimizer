import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()

# Check Groq API Key
#groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )


def create_rag_chain(vector_db):
    """
    Creates a RAG chain using the Chroma DB
    generated from the uploaded XML.
    """

    # Retriever
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 5}
    )

    # Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
You are a Process Optimization Expert.

Analyze the process data provided in the context.

Identify:

1. Bottlenecks
2. Costly activities
3. Resource conflicts
4. Optimization opportunities
5. Expected improvements

Context:
{context}

Question:
{question}
"""
    )

    # Convert documents into text
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
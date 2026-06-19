import streamlit as st

from create_embeddings import create_vector_db
from rag_chain import create_rag_chain


# -----------------------------
# Page Title
# -----------------------------

st.set_page_config(
    page_title="Process Optimization Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Process Optimization Assistant")
st.write(
    "Upload a SIMPLACE XML file and ask questions about process optimization."
)


# -----------------------------
# Session State Initialization
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False


# -----------------------------
# XML Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload XML File",
    type=["xml"]
)


# -----------------------------
# Process XML
# -----------------------------

if uploaded_file is not None and not st.session_state.file_processed:

    with st.spinner("Processing XML and creating embeddings..."):

        db = create_vector_db(uploaded_file)

        st.session_state.rag_chain = create_rag_chain(db)

        st.session_state.file_processed = True

    st.success("✅ XML processed successfully!")


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask a question about the uploaded process..."
)


# -----------------------------
# Ask LLM
# -----------------------------

if question:

    if st.session_state.rag_chain is None:

        st.warning("⚠️ Please upload an XML file first.")

    else:

        # User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # AI Response
        with st.spinner("Analyzing process..."):

            answer = st.session_state.rag_chain.invoke(
                question
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)
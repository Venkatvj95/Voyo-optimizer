import streamlit as st

from create_embeddings import create_vector_db
from rag_chain import create_rag_chain


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Process Optimization Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Process Optimization Assistant")

st.write(
    "Upload a process XML file and ask optimization questions."
)


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Reset Button
# -----------------------------

if st.button("🔄 Reset Session"):

    st.session_state.messages = []

    st.session_state.rag_chain = None

    st.session_state.current_file = None

    st.rerun()

# -----------------------------
# Upload XML
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload XML File",
    type=["xml"]
)


# -----------------------------
# Process New XML
# -----------------------------

if uploaded_file is not None:

    # Process only when a different file is uploaded
    if st.session_state.current_file != uploaded_file.name:

        with st.spinner("Processing XML..."):

            try:

                db = create_vector_db(uploaded_file)

                st.session_state.rag_chain = create_rag_chain(db)

                # Remember current file
                st.session_state.current_file = uploaded_file.name

                # Clear previous chat
                st.session_state.messages = []

                st.success(
                    f"✅ Processed: {uploaded_file.name}"
                )

            except Exception as e:

                st.error(
                    f"Error processing XML: {str(e)}"
                )

                st.stop()


# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(
            message["content"]
        )


# -----------------------------
# User Question
# -----------------------------

question = st.chat_input(
    "Ask a question about the uploaded process..."
)


# -----------------------------
# Ask AI
# -----------------------------

if question:

    if st.session_state.rag_chain is None:

        st.warning(
            "Please upload an XML file first."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.write(question)

        with st.spinner(
            "Analyzing process..."
        ):

            answer = (
                st.session_state
                .rag_chain
                .invoke(question)
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message(
            "assistant"
        ):

            st.write(answer)
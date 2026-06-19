import xml.etree.ElementTree as ET

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma


def create_vector_db(uploaded_file):
    """
    uploaded_file comes from Streamlit file_uploader
    """

    # Parse uploaded XML
    tree = ET.parse(uploaded_file)
    root = tree.getroot()

    docs = []

    for activity in root.findall("Activity"):

        text = f"""
        Activity: {activity.find('Name').text}

        Duration:
        {activity.find('Duration').text}

        Cost:
        {activity.find('Cost').text}

        Resource:
        {activity.find('Resource').text}
        """

        docs.append(
            Document(page_content=text)
        )

    # Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding
    )

    return db
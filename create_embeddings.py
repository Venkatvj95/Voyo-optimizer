import xml.etree.ElementTree as ET

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_db(uploaded_file):

    tree = ET.parse(uploaded_file)
    root = tree.getroot()

    print("Root tag:", root.tag)

    activities = root.findall(".//Activity")

    print("Activities found:", len(activities))

    docs = []

    for activity in activities:

        name = activity.findtext("Name", "")
        duration = activity.findtext("Duration", "")
        cost = activity.findtext("Cost", "")
        resource = activity.findtext("Resource", "")

        text = f"""
        Activity: {name}
        Duration: {duration}
        Cost: {cost}
        Resource: {resource}
        """

        docs.append(
            Document(page_content=text)
        )

    print("Documents created:", len(docs))

    if len(docs) == 0:
        raise ValueError(
            "No Activity records found in XML."
        )

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Docs count:", len(docs))
    db = Chroma.from_documents(
        
        documents=docs,
        embedding=embedding
    )

    return db
from typing import IO

import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.vectorstore import VectorStore
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)


st.title("DocumentGPT")


st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about your files.

    """
)


def embed_file(file: IO[bytes]) -> VectorStore:
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as fp:
        fp.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/sample.docx")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


if file := st.file_uploader(
    "Upload a file (.txt, .pdf or .docx)",
    type=["pdf", "docx", "txt"],
):
    st.write(file)

    # docs = retriever.invoke("ministry of truth")
    # st.write(docs)

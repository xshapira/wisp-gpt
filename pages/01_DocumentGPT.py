import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
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

if file := st.file_uploader(
    "Upload a file (.txt, .pdf or .docx)",
    type=["pdf", "docx", "txt"],
):
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_content, file_path)

    with open(file_path, "wb") as fp:
        fp.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    # llm = ChatOpenAI(
    #     temperature=0.1,
    # )

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
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke("ministry of truth")
    st.write(docs)

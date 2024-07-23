import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


class Embedder:
    @classmethod
    @st.cache_resource(show_spinner="Embedding file...")
    def embed_file(
        cls,
        file,
        _file_path,
        _cache_dir,
        _embeddings=None,
    ):
        file_content = file.read()
        file_path = f"./.cache/{_file_path}/{file.name}"

        with open(file_path, "wb") as fp:
            fp.write(file_content)

        cache_dir = LocalFileStore(f"./.cache/{_cache_dir}/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = _embeddings
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()  # type: ignore[no-any-return]

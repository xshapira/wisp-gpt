import json
from operator import itemgetter
from pathlib import Path

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


class ChatSessionManager:
    def __init__(self):
        self.messages = []
        self.chat_history = []

    def save_message(self, message, role):
        self.messages.append({"message": message, "role": role})

    def send_message(self, message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            self.save_message(message, role)

    def display_chat_history(self):
        for message in self.messages:
            self.send_message(message["message"], message["role"], save=False)

    def save_history(_self, input, output):
        _self.chat_history.append(
            {"input": input, "output": output},
        )

    def save_history_to_file(self, history_file_path):
        history = st.session_state["memory"].chat_memory.messages
        history = messages_to_dict(history)

        with open(history_file_path, "w") as fp:
            json.dump(history, fp, indent=2)

    def restore_history_from_memory(self):
        for history in self.chat_history:
            st.session_state["memory"].save_context(
                {"input": history["input"]},
                {"output": history["output"]},
            )

    @st.cache_data(show_spinner="Loading history from file...")
    def load_history_from_file(_self, history_file_path):
        loaded_message = load_json(history_file_path)
        history = messages_from_dict(loaded_message)
        st.session_state["memory"].chat_memory.messages = history


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        chat_session_manager = ChatSessionManager()
        chat_session_manager.save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


class ChatModel:
    def __init__(
        self,
        llm=None,
        prompt=None,
        memory_llm=None,
        **kwargs,
    ):
        self.llm = llm
        self.prompt = prompt

    def configure_chat_memory(self, memory_llm, **kwargs):
        self.memory_llm = memory_llm

        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationSummaryBufferMemory(
                llm=self.memory_llm,
                max_token_limit=120,
                memory_key="chat_history",
                return_messages=True,
            )


class Embedder:
    @classmethod
    @st.cache_data(show_spinner="Embedding file...")
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


class ChatSession:
    def __init__(self):
        self.chat_session_manager = ChatSessionManager()

    def start(
        self,
        file,
        prompt,
        llm,
        history_file_path,
        **kwargs,
    ):
        retriever = Embedder.embed_file(file, **kwargs)
        self.chat_session_manager.send_message(
            "I'm ready! Ask away.",
            "ai",
            save=False,
        )
        self.chat_session_manager.restore_history_from_memory()
        self.chat_session_manager.display_chat_history()

        message = st.chat_input("Ask me any question about your file")
        if message:
            self.chat_session_manager.send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnablePassthrough.assign(
                    chat_history=RunnableLambda(
                        st.session_state["memory"].load_memory_variables
                    )
                    | itemgetter("chat_history")
                )
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                result = chain.invoke(message)
                self.chat_session_manager.save_history(message, result.content)

            if len(st.session_state["memory"].chat_memory.messages) != 0:
                self.chat_session_manager.save_history_to_file(history_file_path)


def load_txt(path):
    with open(path) as fp:
        return fp.read()


def load_markdown(path):
    with open(path) as fp:
        return fp.read()


def load_json(path):
    with open(path) as fp:
        return json.load(fp)


def format_docs(docs):
    return "/n/n".join(document.page_content for document in docs)


def intro(
    page_title,
    page_icon,
    title,
    markdown,
    history_file_path,
    prompt,
    llm,
    chat_session_args,
):
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
    )

    st.title(title)
    st.markdown(markdown)

    chat_session_manager = ChatSessionManager()

    with st.sidebar:
        file = st.file_uploader(
            "Upload a file (.txt, .pdf or .docx)",
            type=["pdf", "docx", "txt"],
        )
        history_file_path = Path(history_file_path)
        if history_file_path.exists():
            chat_session_manager.load_history_from_file(history_file_path)

    if file:
        chat_session = ChatSession()
        chat_session.start(
            file,
            prompt,
            llm,
            history_file_path,
            **chat_session_args,
        )
    else:
        chat_session_manager.messages = []
        chat_session_manager.chat_history = []


if __name__ == "__main__":
    intro()

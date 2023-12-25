from pathlib import Path

import streamlit as st

from src.chat_session import manage_chat_session
from src.chat_session_manager import load_history_from_file


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

    with st.sidebar:
        file = st.file_uploader(
            "Upload a file (.txt, .pdf or .docx)",
            type=["pdf", "docx", "txt"],
        )
        history_file_path = Path(history_file_path)
        if history_file_path.exists():
            load_history_from_file(history_file_path)

    if file:
        manage_chat_session(
            file=file,
            prompt=prompt,
            llm=llm,
            history_file_path=history_file_path,
            **chat_session_args,
        )
    else:
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []

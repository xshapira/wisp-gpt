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
    """
    Sets up the page configuration, displays a title and markdown content. It allows file uploading, and interaction with the chat.

    Args:
        page_title: The title of the web page
        page_icon: The icon that will be displayed in the browser tab for the page. It should be a URL or a file path to an image file (e.g., a .png or .ico file) that represents the icon.
        title: The title of the page or application
        markdown: A string containing the Markdown content to be displayed on the page.
        history_file_path: The file path where the chat history will be saved.
        prompt: The initial message or question that will be displayed in the chat interface.
        llm: The language model to be used for the chat session.
        chat_session_args: A dictionary containing additional arguments for the `manage_chat_session` function. These arguments are passed to the function when it is called.
    """

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
        history_file = Path(history_file_path)
        if history_file.exists():
            load_history_from_file(history_file)

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

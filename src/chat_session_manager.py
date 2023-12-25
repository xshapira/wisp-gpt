import json

import streamlit as st
from langchain.schema import messages_from_dict, messages_to_dict

from src.utils import load_json


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

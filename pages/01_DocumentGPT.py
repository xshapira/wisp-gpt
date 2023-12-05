import time

import streamlit as st

st.title("DocumentGPT")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message: str, role: str, save: bool = True) -> None:
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)


if message := st.chat_input("send a message to the ai"):
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

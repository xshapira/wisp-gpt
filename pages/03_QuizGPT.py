import json
from pathlib import Path

import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter

from src.chat_session import format_docs
from src.utils import load_file

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

quiz_schema_filename = Path("./prompt_templates/quiz_gpt/quiz_schema.json")
quiz_schema = load_file(quiz_schema_filename)

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
).bind(
    function_call={"name": "generate_quiz"},
    functions=[quiz_schema],
)

questions_prompt_message = [
    SystemMessagePromptTemplate.from_template(
        load_file("./prompt_templates/quiz_gpt/system_message.txt")
    ),
]
questions_prompt = ChatPromptTemplate.from_messages(questions_prompt_message)
questions_chain = {"context": format_docs} | questions_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as fp:
        fp.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)


@st.cache_data(show_spinner="Creating quiz...")
def run_quiz_chain(_docs, topic):
    return questions_chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=3)
    return retriever.get_relevant_documents(term)


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose your preferred way:",
        ("File", "Wikipedia Article"),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf", type=["docx", "txt", "pdf"]
        )
        if file:
            docs = split_file(file)

    elif topic := st.text_input("Search Wikipedia..."):
        docs = wiki_search(topic)


if not docs:
    markdown_file = load_file("./markdowns/quiz_gpt.md")
    st.markdown(markdown_file)
else:
    response = run_quiz_chain(docs, topic or file.name)
    function_call_arguments = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(function_call_arguments)

    with st.form("questions_form"):
        for index, question in enumerate(response["questions"], 1):
            st.write(f"{index}.", question["question"])
            value = st.radio(
                "Select your answer",
                [answer["answer"] for answer in question["answers"]],
                key=f"{index}_radio",
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct")
            elif value is not None:
                st.error("Wrong")
        button = st.form_submit_button("Submit")

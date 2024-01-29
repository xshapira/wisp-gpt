import re

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from src.chat_model import ChatCallbackHandler
from src.chat_session import display_chat_history, send_message
from src.utils import load_file


class ChatModel:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )

        self.answers_messages = [
            SystemMessagePromptTemplate.from_template(
                load_file("./prompt_templates/site_gpt/system_message_answers.txt")
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

        self.choose_messages = [
            SystemMessagePromptTemplate.from_template(
                load_file("./prompt_templates/site_gpt/system_message.txt")
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

        self.answers_prompt = ChatPromptTemplate.from_messages(
            messages=self.answers_messages
        )
        self.choose_prompt = ChatPromptTemplate.from_messages(
            messages=self.choose_messages
        )


chat_model = ChatModel()


def get_answers(input):
    docs = input["context"]
    question = input["question"]
    answers_chain = chat_model.answers_prompt | chat_model.llm
    result = {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }
    return result


def choose_answer(input):
    answers = input["answers"]
    question = input["question"]
    choose_chain = chat_model.choose_prompt | chat_model.llm
    shortened = "\n\n".join(
        f"{answer['answer']}\n\nSource: {answer['source']}\n\n(Date: {answer['date']})\n\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": shortened,
        },
    )


def parse_page(soup):
    """
    Currently using this to scrape the OpenAI site map.
    """
    header = soup.find("header")
    footer = soup.find("footer")
    author_articles_regex = r"Authors(\w+\s\w*View all articles)+"
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    text = (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", " ")
    )
    return re.sub(author_articles_regex, " ", text)


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        # filter_urls=[
        #     r"^(.*\/blog\/).*",
        # ],
        parsing_function=parse_page,
    )
    # Set slower request rate to prevent blocking
    # Default is 2 requests per second
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


def run_site_gpt():
    with st.sidebar:
        url = st.text_input(
            "Enter a URL:",
            placeholder="https://example.com",
        )

    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("The URL must end with .xml")
        else:
            retriever = load_website(url)
            send_message("I'm ready! Ask away.", "ai", save=False)
            # restore history goes here...
            display_chat_history()

            query = st.chat_input("Ask a question about the content of the website.")
            if query:
                send_message(query, "human")
                chain = (
                    {
                        "context": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                with st.chat_message("ai"):
                    result = chain.invoke(query)
                    st.markdown(result.content.replace("$", "\$"))  # noqa: W605
    else:
        st.session_state["messages"] = []


def intro(markdown):
    st.set_page_config(
        page_title="SiteGPT",
        page_icon="🖥️",
    )
    st.markdown(markdown)


def main() -> None:
    markdown_file = load_file("./markdowns/site_gpt.md")
    intro(markdown_file)
    run_site_gpt()


if __name__ == "__main__":
    main()

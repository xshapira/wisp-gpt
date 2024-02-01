import re
from operator import itemgetter
from pathlib import Path

import numpy as np
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chat_model import ChatCallbackHandler
from src.chat_session import display_chat_history, send_message
from src.chat_session_manager import (
    load_history_from_file,
    restore_history_from_memory,
    save_history,
    save_history_to_file,
)
from src.utils import load_file


class ChatModel:
    def __init__(self):
        self.answers_llm = ChatOpenAI(
            temperature=0.1,
        )
        self.choice_llm = ChatOpenAI(
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

    def configure_chat_memory(self):
        self.memory_llm = ChatOpenAI(
            temperature=0.1,
        )

        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationSummaryBufferMemory(
                llm=self.memory_llm,
                max_token_limit=120,
                memory_key="chat_history",
                return_messages=True,
            )


chat_model = ChatModel()


def get_answers(input):
    docs = input["context"]
    question = input["question"]
    answers_chain = chat_model.answers_prompt | chat_model.answers_llm
    result = {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content.replace("$", r"\$"),
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
    choose_chain = chat_model.choose_prompt | chat_model.choice_llm
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


def filter_similar_questions_quickly(query, stored_questions, top_k=10):
    """
    Narrows down the potential matches from a large list of stored questions
    to a smaller subset that shares common words with the user query.

    Args:
        query: The user's question
        stored_questions: The stored questions that we want to compare the user query against
        top_k: The number of similar questions that we want to retrieve. Defaults to 10, but you can change it to any positive integer value.

    Returns:
        A list of the top most similar stored questions to the user query.
    """
    vectorizer = CountVectorizer()
    # vectorize the user query
    query_vector = vectorizer.fit_transform([query])
    # vectorize the stored questions
    stored_vectors = vectorizer.transform(stored_questions)
    # calculate cosine similarities between the user query
    # and the stored questions
    similarities = cosine_similarity(query_vector, stored_vectors)
    # get the top 10 most similar questions
    top_similar_questions = similarities.argsort()[-top_k:][::-1]
    # return the top most similar stored questions
    return [stored_questions[q] for q in top_similar_questions]


def find_most_semantic_match(query, filtered_questions):
    """
    Refines the filtered questions to identify the one that is most semantically similar to the user query.

    Args:
        query: The user's question
        filtered_questions: A list of questions that have been filtered or narrowed down from a larger set of questions. These filtered questions are the ones that will be compared to the user's query to find the most semantically similar match.

    Returns:
        The most similar question from the `filtered_questions` list if the similarity score exceeds a threshold of 0.8. If none of the
        similarities exceed the threshold, returns `None`.
    """
    embedder = OpenAIEmbeddings()
    # generate embedding for the user query
    query_embedding = embedder.encode(query)
    # generate embeddings for filtered questions
    filtered_embeddings = np.array(
        [embedder.encode(question) for question in filtered_questions]
    )
    # calculate cosine similarities between the query embedding
    # each question embedding
    similarities = cosine_similarity([query_embedding], filtered_embeddings).flatten()

    # identify the index of the most similar question
    max_index = np.argmax(similarities)
    # return the most similar question if similarity exceeds a threshold
    return filtered_questions[max_index] if similarities[max_index] > 0.8 else None


def manage_chat_session(url, history_file_path):
    retriever = load_website(url)
    send_message("I'm ready! Ask away.", "ai", save=False)
    restore_history_from_memory()
    display_chat_history()

    query = st.chat_input("Ask a question about the content of the website.")
    if query:
        send_message(query, "human")
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(
                chat_history=RunnableLambda(
                    st.session_state["memory"].load_memory_variables
                )
                | itemgetter("chat_history")
            )
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )

        with st.chat_message("ai"):
            result = chain.invoke(query)
            save_history(query, result.content)

        # checking if the chat memory contains any messages. If it does,
        # then we save the chat history to a file.
        if len(st.session_state["memory"].chat_memory.messages) != 0:
            save_history_to_file(history_file_path)


def run_site_gpt(history_file):
    with st.sidebar:
        url = st.text_input(
            "Enter a URL:",
            placeholder="https://example.com",
        )
        if history_file.exists():
            load_history_from_file(history_file)

    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("The URL must end with .xml")
        else:
            manage_chat_session(url, history_file_path=history_file)
    else:
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []


def intro(markdown):
    st.set_page_config(
        page_title="SiteGPT",
        page_icon="🖥️",
    )
    st.markdown(markdown)


def main() -> None:
    markdown_file = load_file("./markdowns/site_gpt.md")
    history_file = Path("./.cache/chat_questions_history/history.json")
    intro(markdown_file)
    chat_model.configure_chat_memory()
    run_site_gpt(history_file)


if __name__ == "__main__":
    main()

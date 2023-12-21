from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from utils import (
    ChatCallbackHandler,
    ChatModel,
    Embedder,
    intro,
    load_markdown,
)


class App:
    def __init__(self):
        self.chat_model = ChatModel()
        self.embedder = Embedder()
        self.chat_model.llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )
        self.chat_model.memory_llm = ChatOpenAI(
            temperature=0.1,
        )
        self.embedder.file_path = "files"
        self.embedder.cache_dir = "embeddings"
        self.embedder.embeddings = OpenAIEmbeddings()

    def run(self):
        self.chat_model.configure_chat_memory(
            self.chat_model.memory_llm,
        )

        intro_config = {
            "page_title": "DocumentGPT",
            "page_icon": "📄",
            "title": "DocumentGPT",
            "markdown": load_markdown("./markdowns/document_gpt.md"),
            "history_file_path": "./.cache/chat_history/history.json",
            "llm": self.chat_model.llm,
            "chat_session_args": (
                self.embedder.file_path,
                self.embedder.cache_dir,
                self.embedder.embeddings,
            ),
        }

        intro(**intro_config)


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()

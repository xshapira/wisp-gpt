import streamlit as st

from src.utils import load_file


def main() -> None:
    st.set_page_config(
        page_title="WispGPT",
        page_icon="🤖",
    )

    markdown = load_file("./markdowns/home.md")
    st.markdown(markdown)


if __name__ == "__main__":
    main()

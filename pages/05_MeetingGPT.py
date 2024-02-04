import glob
import math
import subprocess
from pathlib import Path

import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydub import AudioSegment

from src.utils import load_file

HAS_TRANSCRIPTION = Path("./.cache/podcast.txt").exists()


class ChatModel:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.1,
        )
        self.first_summary_messages = [
            SystemMessagePromptTemplate.from_template(
                load_file("./prompt_templates/meeting_gpt/first_summary.txt")
            )
        ]
        self.first_summary_prompt = ChatPromptTemplate.from_messages(
            messages=self.first_summary_messages
        )
        self.refine_prompt_messages = [
            SystemMessagePromptTemplate.from_template(
                load_file("./prompt_templates/meeting_gpt/refine.txt")
            )
        ]
        self.refine_prompt = ChatPromptTemplate.from_messages(
            messages=self.refine_prompt_messages
        )


chat_model = ChatModel()


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


@st.cache_data()
def extract_audio_from_video(video_path):
    if HAS_TRANSCRIPTION:
        return
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-af",
        "atempo=1.5",
        audio_path,
    ]

    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_dir):
    if HAS_TRANSCRIPTION:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_length = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_length)

    for i in range(chunks):
        start_time = i * chunk_length
        end_time = (i + 1) * chunk_length
        print(f"start: {start_time}, end: {end_time}")
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_dir}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_file(file):
    if HAS_TRANSCRIPTION:
        return
    with open(file, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            "whisper-1",
            audio_file,
        )
    return transcript["text"]


@st.cache_data()
def transcribe_audio_chunks(chunks_dir, destination):
    if HAS_TRANSCRIPTION:
        return
    files = glob.glob(f"{chunks_dir}/*.mp3", recursive=True)
    files.sort()
    transcripts = [transcribe_file(file) for file in files]
    final_transcript = " ".join(transcripts)
    with open(destination, "w") as fp:
        fp.write(final_transcript)


def upload_video():
    """
    Displays a file uploader widget in the sidebar and allow users to upload
    a video file.

    Returns:
        The uploaded video file object if a file is uploaded; otherwise, None.
    """
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mov", "mkv"],
    )
    return video


def process_video(video):
    """
    Processes the uploaded video by reading its content, extracting audio, cutting audio segments, and transcribing the audio to text.

    Args:
        video (UploadedFile): The uploaded video file to process.

    Returns:
        str: The path to the generated transcript file.
    """
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace(".mp4", ".mp3")
        chunks_path = "./.cache/audio_chunks"
        transcript_path = video_path.replace(".mp4", ".txt")
        with open(video_path, "wb") as fp:
            fp.write(video_content)
        status.update(label="Extracting audio from video...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, chunk_size=10, chunks_dir=chunks_path)
        status.update(label="Transcribing audio...")
        transcribe_audio_chunks(
            chunks_dir=chunks_path,
            destination=transcript_path,
        )
    return transcript_path


def display_transcript(transcript_path):
    """
    Displays the content of the transcript file in a tab.

    Args:
        transcript_path (str): The path to the transcript file.
    """
    with open(transcript_path) as file:
        st.write(file.read())


def create_summary(loader):
    """
    Creates a summary for the loaded text from a TextLoader object.

    Args:
        loader (TextLoader): A TextLoader object containing the text to summarize.

    Returns:
        str: The generated summary of the text.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    first_summary_chain = (
        chat_model.first_summary_prompt | chat_model.llm | StrOutputParser()
    )
    summary = first_summary_chain.invoke(
        {
            "text": docs[0].page_content,
        }
    )
    refine_chain = chat_model.refine_prompt | chat_model.llm | StrOutputParser()

    with st.status("Summarizing...") as status:
        for i, doc in enumerate(docs[1:], start=1):
            status.update(
                label=f"Generating summary for {i}/{len(docs) - 1}",
            )
            summary = refine_chain.invoke(
                {
                    "existing_summary": summary,
                    "context": doc.page_content,
                }
            )
            st.write(summary)
        return summary


def generate_summary(transcript_path):
    """
    Generates a summary of the transcript and displays it in a tab. This function initializes the summary generation process and displays the final summary.

    Args:
        transcript_path (str): The path to the transcript file used for summary generation.
    """
    start = st.button("Generate summary")
    if start:
        loader = TextLoader(transcript_path)
        summary = create_summary(loader)
        st.write(summary)


def handle_qa(transcript_path):
    """
    Handles the Q&A functionality by embedding the transcript and retrieving answers to a predefined question.

    Args:
        transcript_path (str): The path to the transcript file used for Q&A.
    """
    retriever = embed_file(transcript_path)
    docs = retriever.invoke("do they talk about marcus aurelius?")
    st.write(docs)


def run_meeting_gpt():
    with st.sidebar:
        video = upload_video()

    if video:
        transcript_path = process_video(video)
        transcript_tab, summary_tab, qa_tab = st.tabs(
            [
                "Transcript",
                "Summary",
                "Q&A",
            ],
        )
        with transcript_tab:
            display_transcript(transcript_path)
        with summary_tab:
            generate_summary(transcript_path)
        with qa_tab:
            handle_qa(transcript_path)


def intro(markdown_file):
    st.set_page_config(
        page_title="MeetingGPT",
        page_icon="💼",
    )
    st.markdown(markdown_file)


def main() -> None:
    markdown_file = load_file("./markdowns/meeting_gpt.md")
    intro(markdown_file)
    run_meeting_gpt()


if __name__ == "__main__":
    main()

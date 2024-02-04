import glob
import math
import subprocess
from pathlib import Path

import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydub import AudioSegment

from src.utils import load_file

llm = ChatOpenAI(
    temperature=0.1,
)

has_transcription = Path("./.cache/podcast.txt").exists()


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcription:
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
    if has_transcription:
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
    if has_transcription:
        return
    with open(file, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            "whisper-1",
            audio_file,
        )
    return transcript["text"]


@st.cache_data()
def transcribe_audio_chunks(chunks_dir, destination):
    if has_transcription:
        return
    files = glob.glob(f"{chunks_dir}/*.mp3", recursive=True)
    files.sort()
    transcripts = [transcribe_file(file) for file in files]
    final_transcript = " ".join(transcripts)
    with open(destination, "w") as fp:
        fp.write(final_transcript)


markdown_file = load_file("./markdowns/meeting_gpt.md")
st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)
st.markdown(markdown_file)


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mov", "mkv"],
    )

if video:
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace(".mp4", ".mp3")
        chunks_dir = "./.cache/audio_chunks"
        transcript_dir = video_path.replace(".mp4", ".txt")
        with open(video_path, "wb") as fp:
            fp.write(video_content)
        status.update(label="Extracting audio from video...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, chunk_size=10, chunks_dir=chunks_dir)
        status.update(label="Transcribing audio...")
        transcribe_audio_chunks(
            chunks_dir=chunks_dir,
            destination=transcript_dir,
        )

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ],
    )

    with transcript_tab:
        with open(transcript_dir) as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader("./.cache/podcast.txt")
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke(
                {
                    "text": docs[0].page_content,
                }
            )
            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )
            refine_chain = refine_prompt | llm | StrOutputParser()

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

import subprocess

import streamlit as st

from src.utils import load_file


def extract_audio_from_video(video_path):
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-af",
        "atempo=1.5",
        audio_path,
    ]

    subprocess.run(command)


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
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        with open(video_path, "wb") as fp:
            fp.write(video_content)
        extract_audio_from_video(video_path)

import json
from pathlib import Path
from typing import overload


@overload
def load_file(path: str) -> str:
    """
    Loads a text file and returns its contents as a string.

    Args:
        file_path (str): The path to the text file to be loaded.

    Returns:
        str: The contents of the text file.
    """
    with open(path) as fp:
        return fp.read()


@overload
def load_file(path: str) -> str:
    """
    Loads a Markdown file and returns its contents as a string.

    Args:
        file_path (str): The path to the Markdown file to be loaded.

    Returns:
        str: The contents of the Markdown file.
    """
    with open(path) as fp:
        return fp.read()


@overload
def load_file(path: Path) -> dict:
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(path) as fp:
        return json.load(fp)


def load_file(path):
    """
    Default implementation of the `load_file` function.
    """
    try:
        with open(path) as fp:
            return fp.read()
    except ValueError as exc:
        raise ValueError("Failed to load file") from exc

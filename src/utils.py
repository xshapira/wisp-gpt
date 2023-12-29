import json
from functools import singledispatch
from pathlib import Path


@singledispatch
def load_file(path):
    """
    Default implementation of the `load_file` function.
    """
    try:
        with open(path) as fp:
            return fp.read()
    except ValueError as exc:
        raise ValueError(f"Failed to load file: {exc}") from exc


@load_file.register
def _(path: str) -> str:
    """
    Loads a text or Markdown file and returns its contents as a string.

    Args:
        file_path (str): The path to the text file to be loaded.

    Returns:
        str: The contents of the text file.
    """
    with open(path) as fp:
        return fp.read()


@load_file.register
def _(path: Path) -> dict:
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(path) as fp:
        return json.load(fp)

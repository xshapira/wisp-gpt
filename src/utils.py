import json


def load_txt(path):
    """
    Loads a text file and returns its contents as a string.

    Args:
        file_path (str): The path to the text file to be loaded.

    Returns:
        str: The contents of the text file.
    """
    with open(path) as fp:
        return fp.read()


def load_markdown(path):
    """
    Loads a Markdown file and returns its contents as a string.

    Args:
        file_path (str): The path to the Markdown file to be loaded.

    Returns:
        str: The contents of the Markdown file.
    """
    with open(path) as fp:
        return fp.read()


def load_json(path):
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(path) as fp:
        return json.load(fp)

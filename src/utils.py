import json


def load_txt(path):
    with open(path) as fp:
        return fp.read()


def load_markdown(path):
    with open(path) as fp:
        return fp.read()


def load_json(path):
    with open(path) as fp:
        return json.load(fp)

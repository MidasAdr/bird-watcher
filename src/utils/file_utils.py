import re


def safe_filename(text):

    text = text.replace(" ", "_")

    return re.sub(r"[^A-Za-z0-9_\-]", "", text)
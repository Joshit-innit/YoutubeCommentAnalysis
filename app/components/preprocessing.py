import re

from nltk.corpus import stopwords

stop_words = set(
    stopwords.words("english")
)

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)
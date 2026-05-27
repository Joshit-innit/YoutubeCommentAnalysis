import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Load dataset
df = pd.read_csv("data/raw/comments.csv")

# Stopwords
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove punctuation/special chars
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenization
    words = text.split()

    # Remove stopwords + stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Apply preprocessing
df["clean_comment"] = df["comment"].astype(str).apply(clean_text)

# Save processed dataset
df.to_csv(
    "data/processed/cleaned_comments.csv",
    index=False
)

print(df[["comment", "clean_comment"]].head())
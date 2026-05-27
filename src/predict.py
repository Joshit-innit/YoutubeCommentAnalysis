import pandas as pd
import re
import joblib

from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("models/toxicity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load comments
df = pd.read_csv("data/raw/comments.csv")

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Clean comments
df["clean_comment"] = df["comment"].apply(clean_text)

# Transform text
X = vectorizer.transform(df["clean_comment"])

# Predict toxicity
df["toxicity_prediction"] = model.predict(X)

# Save predictions
df.to_csv(
    "data/processed/predicted_comments.csv",
    index=False
)

# Toxic comment count
toxic_count = df["toxicity_prediction"].sum()

# Total comments
total_comments = len(df)

# Toxicity percentage
toxicity_percentage = (
    toxic_count / total_comments
) * 100

print("\n========= RESULTS =========")

print(f"Total Comments: {total_comments}")

print(f"Toxic Comments: {toxic_count}")

print(f"Toxicity Percentage: {toxicity_percentage:.2f}%")

print("\nTop Toxic Comments:\n")

# Show toxic comments
toxic_comments = df[
    df["toxicity_prediction"] == 1
]

print(
    toxic_comments[
        ["author", "comment"]
    ].head(10)
)
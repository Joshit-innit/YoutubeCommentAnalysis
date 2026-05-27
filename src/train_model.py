import pandas as pd
import re
import joblib

from nltk.corpus import stopwords


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/raw/train.csv").sample(
    20000,
    random_state=42
)

# Keep only required columns
df = df[["comment_text", "toxic"]]

# Remove missing values
df.dropna(inplace=True)

# Stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenization
    words = text.split()

    # Remove stopwords + stemming
    words = [
    word
    for word in words
    if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
df["clean_text"] = df["comment_text"].apply(clean_text)

# Features and target
X = df["clean_text"]
y = df["toxic"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, "models/toxicity_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
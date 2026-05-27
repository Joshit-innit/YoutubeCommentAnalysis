import joblib

model = joblib.load(
    "models/toxicity_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

def predict_toxicity(clean_comments):

    X = vectorizer.transform(
        clean_comments
    )

    predictions = model.predict(X)

    return predictions
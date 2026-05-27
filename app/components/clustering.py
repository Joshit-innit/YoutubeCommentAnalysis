# =========================================================
# 1️⃣ CREATE FILE:
# app/components/clustering.py
# =========================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import pandas as pd

# =========================================================
# COMMENT CLUSTERING
# =========================================================

def cluster_comments(
    comments,
    n_clusters=4
):

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    X = vectorizer.fit_transform(
        comments
    )

    # KMeans Clustering
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    model.fit(X)

    # Assign cluster labels
    clusters = model.labels_

    # Extract keywords for each cluster
    terms = vectorizer.get_feature_names_out()

    cluster_keywords = {}

    for i in range(n_clusters):

        center_terms = (
            model.cluster_centers_[i]
            .argsort()[-5:]
        )

        keywords = [
            terms[index]
            for index in center_terms
        ]

        cluster_keywords[i] = keywords

    return clusters, cluster_keywords
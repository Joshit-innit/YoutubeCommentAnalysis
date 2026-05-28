from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_comments(comments, n_clusters=4):
    cleaned_comments = [
        str(comment).strip()
        for comment in comments
        if str(comment).strip()
    ]

    if not cleaned_comments:
        return [0 for _ in comments], {0: ["no", "topic", "signal"]}

    cluster_count = min(n_clusters, len(cleaned_comments))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
    )

    try:
        matrix = vectorizer.fit_transform(cleaned_comments)
    except ValueError:
        return [0 for _ in comments], {0: ["low", "text", "signal"]}

    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    )

    model.fit(matrix)

    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster_id in range(cluster_count):
        center_terms = model.cluster_centers_[cluster_id].argsort()[-5:][::-1]
        cluster_keywords[cluster_id] = [
            terms[index]
            for index in center_terms
        ]

    labels = []
    label_index = 0

    for comment in comments:
        if str(comment).strip():
            labels.append(int(model.labels_[label_index]))
            label_index += 1
        else:
            labels.append(0)

    return labels, cluster_keywords

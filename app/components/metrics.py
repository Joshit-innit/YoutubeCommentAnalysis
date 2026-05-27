def calculate_metrics(df):

    total_comments = len(df)

    toxic_comments = (
        df["toxicity_prediction"]
        .sum()
    )

    toxicity_percentage = (
        toxic_comments
        / total_comments
    ) * 100

    positive_count = (
        df["sentiment"]
        == "Positive"
    ).sum()

    negative_count = (
        df["sentiment"]
        == "Negative"
    ).sum()

    positive_percentage = (
        positive_count
        / total_comments
    ) * 100

    negative_percentage = (
        negative_count
        / total_comments
    ) * 100

    satisfaction_score = (
        positive_percentage
        - negative_percentage
    )

    return {
        "total_comments": total_comments,
        "toxicity_percentage":
            toxicity_percentage,
        "positive_percentage":
            positive_percentage,
        "negative_percentage":
            negative_percentage,
        "satisfaction_score":
            satisfaction_score
    }
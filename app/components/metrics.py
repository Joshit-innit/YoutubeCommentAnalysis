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

    engagement_score = min(
        100,
        (
            (df["likes"].mean() * 2.2)
            + (df["comment_length"].mean() * 0.45)
            + (positive_percentage * 0.35)
        ),
    )

    viewer_mood = (
        (positive_percentage * 1.1)
        - (negative_percentage * 0.7)
        - (toxicity_percentage * 0.35)
    )
    viewer_mood = max(-100, min(100, viewer_mood))

    controversy_score = min(100, min(positive_percentage, negative_percentage) * 2)

    viral_potential = min(
        100,
        (
            (engagement_score * 0.52)
            + (100 - toxicity_percentage) * 0.14
            + (100 - abs(positive_percentage - negative_percentage)) * 0.34
        ),
    )

    audience_loyalty = max(
        0,
        min(
            100,
            (positive_percentage * 0.68)
            + (engagement_score * 0.32)
            - (toxicity_percentage * 0.18),
        ),
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
            satisfaction_score,
        "engagement_score":
            engagement_score,
        "viewer_mood_index":
            viewer_mood,
        "controversy_score":
            controversy_score,
        "viral_potential_score":
            viral_potential,
        "audience_loyalty_score":
            audience_loyalty,
    }
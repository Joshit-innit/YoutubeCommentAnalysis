def generate_insights(
    positive_percentage,
    negative_percentage,
    toxicity_percentage,
    spam_percentage,
    satisfaction_score
):

    insights = []

    if positive_percentage > 60:

        insights.append(
            "Audience reaction is highly positive."
        )

    if toxicity_percentage > 20:

        insights.append(
            "High toxicity detected in comments."
        )

    if spam_percentage > 5:

        insights.append(
            "Suspicious spam activity detected."
        )

    if negative_percentage > 40:

        insights.append(
            "Audience dissatisfaction is high."
        )

    if satisfaction_score > 40:

        insights.append(
            "Viewer satisfaction appears strong."
        )

    if len(insights) == 0:

        insights.append(
            "Audience reactions appear balanced."
        )

    return insights
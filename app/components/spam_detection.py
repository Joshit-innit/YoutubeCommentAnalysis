def detect_spam(df):

    repeated_comments = (
        df["comment"]
        .value_counts()
    )

    spam_comments = repeated_comments[
        repeated_comments > 3
    ]

    spam_percentage = (
        len(spam_comments)
        / len(df)
    ) * 100

    return spam_comments, spam_percentage
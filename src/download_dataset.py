import pandas as pd

url = "https://huggingface.co/datasets/civil_comments/resolve/main/civil_comments.csv"

# Read dataset
df = pd.read_csv(url)

# Keep required columns
df = df[["text", "toxicity"]]

# Convert to binary labels
df["toxic"] = df["toxicity"].apply(
    lambda x: 1 if x >= 0.5 else 0
)

# Rename column
df.rename(columns={"text": "comment_text"}, inplace=True)

# Drop original toxicity column
df.drop("toxicity", axis=1, inplace=True)

# Save dataset
df.to_csv("data/raw/train.csv", index=False)

print(df.head())

print(f"\nDataset downloaded successfully!")
print(f"Rows: {len(df)}")
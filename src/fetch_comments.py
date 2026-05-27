import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

# Build YouTube API client
youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

# Extract video ID from URL
def extract_video_id(url):

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None


# Fetch comments
def fetch_comments(video_id):

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    response = request.execute()

    while request:

        for item in response["items"]:

            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "author": snippet["authorDisplayName"],
                "comment": snippet["textDisplay"],
                "likes": snippet["likeCount"],
                "published_at": snippet["publishedAt"]
            })

        # Pagination
        if "nextPageToken" in response:

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response["nextPageToken"],
                maxResults=100,
                textFormat="plainText"
            )

            response = request.execute()

        else:
            break

    return pd.DataFrame(comments)


# Main execution
if __name__ == "__main__":

    video_url = input("Enter YouTube Video URL: ")

    video_id = extract_video_id(video_url)

    if not video_id:
        print("Invalid URL")
        exit()

    df = fetch_comments(video_id)

    # Save comments
    df.to_csv("data/raw/comments.csv", index=False)

    print(df.head())

    print(f"\nSuccessfully fetched {len(df)} comments!")
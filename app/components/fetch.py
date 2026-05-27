import pandas as pd
import re

from googleapiclient.discovery import build

# =====================================
# EXTRACT VIDEO ID
# =====================================

def extract_video_id(url):

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None

# =====================================
# FETCH COMMENTS
# =====================================

def fetch_comments(video_id, api_key):

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

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

            snippet = item[
                "snippet"
            ]["topLevelComment"]["snippet"]

            comments.append({
                "author": snippet[
                    "authorDisplayName"
                ],
                "comment": snippet[
                    "textDisplay"
                ],
                "likes": snippet[
                    "likeCount"
                ],
                "published_at": snippet[
                    "publishedAt"
                ]
            })

        if "nextPageToken" in response:

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response[
                    "nextPageToken"
                ],
                maxResults=100,
                textFormat="plainText"
            )

            response = request.execute()

        else:
            break

    return pd.DataFrame(comments)
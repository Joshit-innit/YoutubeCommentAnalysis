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


def fetch_video_metadata(video_id, api_key):

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    video_request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    video_response = video_request.execute()

    if not video_response.get("items"):
        return {}

    video_item = video_response["items"][0]
    snippet = video_item.get("snippet", {})
    stats = video_item.get("statistics", {})

    channel_id = snippet.get("channelId")
    subscribers = None

    if channel_id:
        channel_request = youtube.channels().list(
            part="statistics",
            id=channel_id
        )
        channel_response = channel_request.execute()
        if channel_response.get("items"):
            subscribers = channel_response["items"][0].get(
                "statistics",
                {}
            ).get("subscriberCount")

    return {
        "title": snippet.get("title", "Unknown Title"),
        "channel": snippet.get("channelTitle", "Unknown Channel"),
        "published_at": snippet.get("publishedAt", ""),
        "thumbnail": (
            snippet.get("thumbnails", {})
            .get("high", {})
            .get("url", "")
        ),
        "views": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
        "likes": int(stats.get("likeCount", 0)) if stats.get("likeCount") else 0,
        "subscribers": int(subscribers) if subscribers else None,
    }
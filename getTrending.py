# pip install google-api-python-client
# pip install pandas
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime

def get_top_songs(api_key, region_code='RO', max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    trending_videos = []
    next_page_token = None

    while len(trending_videos) < max_results:
        request = youtube.videos().list(
            part='snippet,statistics',
            chart='mostPopular',
            regionCode=region_code,
            videoCategoryId='10',
            maxResults=min(50, max_results - len(trending_videos)),
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            published_at_iso = item['snippet']['publishedAt']
            dt = datetime.strptime(published_at_iso, "%Y-%m-%dT%H:%M:%SZ")
            published_at = dt.strftime("%d-%m-%Y %H:%M:%S")

            video_info = {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'published_at': published_at,
                'view_count': f"Views: {item['statistics']['viewCount']}",
                'like_count': f"Likes: {item['statistics'].get('likeCount', '0')}",
                'comment_count': f"Comments: {item['statistics'].get('commentCount', '0')}"
            }
            trending_videos.append(video_info)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return trending_videos

API_KEY = 'AIzaSyBAy5Ivz35bsUj82k_-vtlgBB-O0TV9_HQ'

top_songs = get_top_songs(API_KEY, max_results=100)

df = pd.DataFrame(top_songs)
df.to_csv('top_romanian_songs.csv', index=False)

with open('top_romanian_songs.txt', 'w', encoding='utf-8') as file:
    for song in top_songs:
        file.write(f"Title: {song['title']}\n")
        file.write(f"Channel: {song['channel']}\n")
        file.write(f"Published at: {song['published_at']}\n")
        file.write(f"{song['view_count']}\n")
        file.write(f"{song['like_count']}\n")
        file.write(f"{song['comment_count']}\n")
        file.write("\n")


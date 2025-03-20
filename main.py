import os
import yt_dlp
import subprocess
import uuid
import pandas as pd

# Configuration
URLS_FILE = 'video_ids.txt'
VIDEOS_DIR = './videos'
SCENES_DIR = './scenes'
METADATA_CSV = 'metadata.csv'

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)


def download_video(video_id):
    # Check if video already exists
    for file in os.listdir(VIDEOS_DIR):
        if file.startswith(video_id) and file.endswith('.mp4'):
            video_path = os.path.join(VIDEOS_DIR, file)
            print(f"Video already exists: {video_path}")
            return video_path

    # If not, download it
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(VIDEOS_DIR, f"{video_id}.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded_files = os.listdir(VIDEOS_DIR)
    for file in downloaded_files:
        if file.startswith(video_id) and file.endswith('.mp4'):
            return os.path.join(VIDEOS_DIR, file)
    return None


def detect_and_split_scenes(video_path, video_id):
    output_dir = os.path.join(SCENES_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    command = [
        'scenedetect', '-i', video_path,
        'detect-adaptive', 'split-video', '-o', output_dir
    ]
    subprocess.run(command, check=True)

    scenes = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.endswith('.mp4')
    ])
    return scenes


def main():
    # Read video IDs from file
    with open(URLS_FILE, 'r') as file:
        video_ids = [line.strip() for line in file if line.strip()]

    metadata = []

    for idx, video_id in enumerate(video_ids):
        print(f"Processing video {idx+1}/{len(video_ids)}: {video_id}")

        # Download the video
        try:
            video_path = download_video(video_id)
            if not video_path:
                print(f"Download failed for video ID: {video_id}")
                continue
            print(f"Downloaded video: {video_path}")
        except Exception as e:
            print(f"Error downloading video ID {video_id}: {e}")
            continue

        # Split into scenes
        try:
            scenes = detect_and_split_scenes(video_path, video_id)
            print(f"Detected {len(scenes)} scenes.")
        except subprocess.CalledProcessError as e:
            print(f"Scene detection failed for {video_path}: {e}")
            continue

        # Store metadata
        for scene_idx, scene_path in enumerate(scenes):
            metadata.append({
                'video_id': video_id,
                'original_url': f"https://www.youtube.com/watch?v={video_id}",
                'video_path': video_path,
                'scene_index': scene_idx,
                'scene_path': scene_path,
            })

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(METADATA_CSV, index=False)
    print(f"Metadata saved to {METADATA_CSV}")


if __name__ == "__main__":
    main()

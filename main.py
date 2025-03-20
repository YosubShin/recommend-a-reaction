import os
import yt_dlp
import subprocess
import uuid
import pandas as pd
import argparse
import json

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Download and split YouTube videos into scenes')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Output directory for all files')
args = parser.parse_args()

# Configuration
URLS_FILE = 'video_ids.txt'
OUTPUT_DIR = args.output_dir
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'videos')
SCENES_DIR = os.path.join(OUTPUT_DIR, 'scenes')
METADATA_CSV = os.path.join(OUTPUT_DIR, 'metadata.csv')
# Directory for scene info JSON files
SCENE_INFO_DIR = os.path.join(OUTPUT_DIR, 'scene_info')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)
os.makedirs(SCENE_INFO_DIR, exist_ok=True)  # Create scene info directory


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

    # Check if scene info already exists
    scene_info_path = os.path.join(SCENE_INFO_DIR, f"{video_id}_scenes.json")
    if os.path.exists(scene_info_path):
        print(
            f"Scene info already exists for {video_id}, loading from {scene_info_path}")
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)

        # Verify all scene files exist
        scenes = scene_info['scenes']
        all_scenes_exist = all(os.path.exists(scene_path)
                               for scene_path in scenes)

        if all_scenes_exist:
            print(f"All {len(scenes)} scenes already exist for {video_id}")
            return scenes
        else:
            print(f"Some scenes are missing, reprocessing {video_id}")

    # If we get here, we need to process the video
    command = [
        'scenedetect', '-i', video_path,
        'detect-adaptive', 'split-video', '-o', output_dir
    ]
    subprocess.run(command, check=True)

    scenes = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.endswith('.mp4')
    ])

    # Save scene info to JSON
    scene_info = {
        'video_id': video_id,
        'video_path': video_path,
        'scenes': scenes,
        'scene_count': len(scenes)
    }

    with open(scene_info_path, 'w') as f:
        json.dump(scene_info, f, indent=2)

    return scenes


def main():
    # Read video IDs from file
    with open(URLS_FILE, 'r') as file:
        video_ids = [line.strip() for line in file if line.strip()]

    # Load existing metadata if it exists
    existing_metadata = []
    if os.path.exists(METADATA_CSV):
        try:
            existing_df = pd.read_csv(METADATA_CSV)
            existing_metadata = existing_df.to_dict('records')
            print(f"Loaded {len(existing_metadata)} existing metadata records")

            # Create a set of already processed video IDs
            processed_video_ids = set(item['video_id']
                                      for item in existing_metadata)
            print(f"Already processed {len(processed_video_ids)} videos")
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
            existing_metadata = []

    metadata = existing_metadata.copy()

    # Track which videos we've already processed in this run
    processed_in_this_run = set()

    for idx, video_id in enumerate(video_ids):
        # Skip if this video is already in our metadata and we've processed all its scenes
        if video_id in processed_in_this_run:
            print(f"Already processed {video_id} in this run, skipping")
            continue

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

        # Check if we already have metadata for this video
        existing_scenes_for_video = [
            item for item in existing_metadata if item['video_id'] == video_id]

        # If we have the same number of scenes, assume it's already processed correctly
        if existing_scenes_for_video and len(existing_scenes_for_video) == len(scenes):
            print(
                f"Metadata for all scenes of {video_id} already exists, skipping")
            processed_in_this_run.add(video_id)
            continue

        # Remove any existing metadata for this video (we'll replace it)
        metadata = [item for item in metadata if item['video_id'] != video_id]

        # Store metadata
        for scene_idx, scene_path in enumerate(scenes):
            metadata.append({
                'video_id': video_id,
                'original_url': f"https://www.youtube.com/watch?v={video_id}",
                'video_path': video_path,
                'scene_index': scene_idx,
                'scene_path': scene_path,
            })

        processed_in_this_run.add(video_id)

        # Save metadata after each video is processed
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(METADATA_CSV, index=False)
        print(f"Updated metadata saved to {METADATA_CSV}")

    # Final save of metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(METADATA_CSV, index=False)
    print(f"Final metadata saved to {METADATA_CSV}")


if __name__ == "__main__":
    main()

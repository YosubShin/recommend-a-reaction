import os
import yt_dlp
import subprocess
import uuid
import pandas as pd
import argparse
import json
import concurrent.futures
import threading
import whisper  # For transcription

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Download, split YouTube videos into scenes, and transcribe them')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Output directory for all files')
parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers for processing videos')
args = parser.parse_args()

# Configuration
URLS_FILE = 'video_ids.txt'
OUTPUT_DIR = args.output_dir
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'videos')
SCENES_DIR = os.path.join(OUTPUT_DIR, 'scenes')
METADATA_CSV = os.path.join(OUTPUT_DIR, 'metadata.csv')
# Directory for scene info JSON files
SCENE_INFO_DIR = os.path.join(OUTPUT_DIR, 'scene_info')
# Directory for transcriptions
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, 'transcripts')
# Number of parallel workers
NUM_WORKERS = args.workers

# Thread-safe lock for metadata updates
metadata_lock = threading.Lock()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)
os.makedirs(SCENE_INFO_DIR, exist_ok=True)  # Create scene info directory
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)  # Create transcripts directory

# Load whisper turbo model
print("Loading Whisper turbo model for transcription...")
whisper_model = whisper.load_model("turbo")


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


def transcribe_scene(scene_path, video_id, scene_idx, language=None):
    """
    Transcribe a single scene using Whisper

    Args:
        scene_path: Path to the scene audio/video file
        video_id: YouTube video ID
        scene_idx: Scene index number
        language: Optional language code (e.g., 'en', 'es', 'fr', etc.)
                 If None, Whisper will auto-detect the language
    """
    # Create video-specific transcript directory
    video_transcript_dir = os.path.join(TRANSCRIPTS_DIR, video_id)
    os.makedirs(video_transcript_dir, exist_ok=True)

    # Define output path for transcript
    transcript_filename = f"scene_{scene_idx}.json"
    transcript_path = os.path.join(video_transcript_dir, transcript_filename)

    # Check if transcript already exists
    if os.path.exists(transcript_path):
        print(f"Transcript already exists for {video_id} scene {scene_idx}")
        with open(transcript_path, 'r') as f:
            return json.load(f)

    try:
        print(f"Transcribing {video_id} scene {scene_idx}...")

        # If language is specified, use it; otherwise let Whisper auto-detect
        if language:
            result = whisper_model.transcribe(scene_path, language=language)
            print(f"Transcribing using specified language: {language}")
        else:
            result = whisper_model.transcribe(scene_path)
            print(
                f"Transcribing with auto-detected language: {result.get('language', 'unknown')}")

        # Save transcript to file
        with open(transcript_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result
    except Exception as e:
        print(f"Error transcribing {scene_path}: {e}")
        return None


def detect_language(video_path):
    """
    Detect the language of the first 30 seconds of a video using Whisper

    Args:
        video_path: Path to the video file

    Returns:
        Detected language code (e.g., 'en', 'es', 'fr', etc.)
    """
    try:
        print(f"Detecting language from first 30 seconds of {video_path}...")
        # Extract first 30 seconds of audio for language detection
        temp_audio_path = os.path.join(
            OUTPUT_DIR, f"temp_audio_{uuid.uuid4()}.wav")

        # Use ffmpeg to extract first 30 seconds
        command = [
            'ffmpeg', '-i', video_path, '-t', '30', '-q:a', '0',
            '-map', 'a', temp_audio_path, '-y'
        ]
        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load audio using Whisper's utility functions
        audio = whisper.load_audio(temp_audio_path)
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(
            audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)

        # Detect language directly without transcription
        _, probs = whisper_model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        print(
            f"Detected language: {detected_language} (probability: {probs[detected_language]:.2%})")
        return detected_language
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None


def process_video(video_id, existing_metadata):
    """Process a single video - download, split into scenes, and transcribe"""
    print(f"Processing video: {video_id}")
    result_metadata = []

    try:
        # Download the video
        video_path = download_video(video_id)
        if not video_path:
            print(f"Download failed for video ID: {video_id}")
            return []
        print(f"Downloaded video: {video_path}")

        # Detect language from the original video
        detected_language = detect_language(video_path)
        print(f"Video {video_id} language detected as: {detected_language}")

        # Split into scenes
        try:
            scenes = detect_and_split_scenes(video_path, video_id)
            print(f"Detected {len(scenes)} scenes for {video_id}.")
        except subprocess.CalledProcessError as e:
            print(f"Scene detection failed for {video_path}: {e}")
            return []

        # Check if we already have metadata for this video
        existing_scenes_for_video = [
            item for item in existing_metadata if item['video_id'] == video_id]

        # Only skip processing if we have the same number of scenes AND all scenes have transcripts
        if (existing_scenes_for_video and
            len(existing_scenes_for_video) == len(scenes) and
            all(item.get('transcript_path') and os.path.exists(item['transcript_path'])
                for item in existing_scenes_for_video)):
            print(
                f"Metadata and transcripts for all scenes of {video_id} already exist, using existing")
            return existing_scenes_for_video

        print(f"Need to process transcripts for {video_id}")

        # Store metadata
        for scene_idx, scene_path in enumerate(scenes):
            # Transcribe the scene
            transcript = transcribe_scene(
                scene_path, video_id, scene_idx, detected_language)

            metadata_entry = {
                'video_id': video_id,
                'original_url': f"https://www.youtube.com/watch?v={video_id}",
                'video_path': video_path,
                'scene_index': scene_idx,
                'scene_path': scene_path,
            }

            # Add transcript information
            if transcript:
                transcript_path = os.path.join(
                    TRANSCRIPTS_DIR, video_id, f"scene_{scene_idx}.json")
                metadata_entry['transcript_path'] = transcript_path
                metadata_entry['transcript_text'] = transcript.get('text', '')
            else:
                metadata_entry['transcript_path'] = None
                metadata_entry['transcript_text'] = ''

            result_metadata.append(metadata_entry)

        return result_metadata
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        return []


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

            # Create a set of already processed video IDs with complete processing
            processed_video_ids = set()
            for video_id in set(item['video_id'] for item in existing_metadata):
                # Check if all scenes for this video have transcripts
                scenes_for_video = [
                    item for item in existing_metadata if item['video_id'] == video_id]
                if all(item.get('transcript_path') and os.path.exists(item['transcript_path']) for item in scenes_for_video):
                    processed_video_ids.add(video_id)

            print(f"Already fully processed {len(processed_video_ids)} videos")
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
            existing_metadata = []

    # Filter out videos that have already been fully processed
    videos_to_process = [v for v in video_ids if v not in processed_video_ids]
    print(
        f"Processing {len(videos_to_process)} videos with {NUM_WORKERS} workers")

    # Process videos in parallel
    all_metadata = existing_metadata.copy()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all video processing tasks
        future_to_video = {
            executor.submit(process_video, video_id, existing_metadata): video_id
            for video_id in videos_to_process
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                new_metadata = future.result()
                if new_metadata:
                    with metadata_lock:
                        # Remove any existing metadata for this video
                        all_metadata = [
                            item for item in all_metadata if item['video_id'] != video_id]
                        # Add new metadata
                        all_metadata.extend(new_metadata)
                        # Save intermediate results
                        metadata_df = pd.DataFrame(all_metadata)
                        metadata_df.to_csv(METADATA_CSV, index=False)
                        print(
                            f"Updated metadata for {video_id} saved to {METADATA_CSV}")
            except Exception as e:
                print(f"Error processing results for {video_id}: {e}")

    # Final save of metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(METADATA_CSV, index=False)
    print(f"Final metadata saved to {METADATA_CSV}")


if __name__ == "__main__":
    main()

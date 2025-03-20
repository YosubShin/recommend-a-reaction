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
import cv2
from ultralytics import YOLO  # For face detection
import numpy as np
from PIL import Image
import time
import av  # Add PyAV import

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
# Directory for face images
FACES_DIR = os.path.join(OUTPUT_DIR, 'faces')
# Directory for face metadata
FACE_METADATA_DIR = os.path.join(OUTPUT_DIR, 'face_metadata')
# Directory for consolidated scene data
CONSOLIDATED_SCENE_DATA_DIR = os.path.join(OUTPUT_DIR, 'scenes')
# Number of parallel workers
NUM_WORKERS = args.workers

# Thread-safe lock for metadata updates
metadata_lock = threading.Lock()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)
os.makedirs(SCENE_INFO_DIR, exist_ok=True)  # Create scene info directory
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)  # Create transcripts directory
os.makedirs(FACES_DIR, exist_ok=True)  # Create faces directory
os.makedirs(FACE_METADATA_DIR, exist_ok=True)  # Create face metadata directory

# Load whisper turbo model
print("Loading Whisper turbo model for transcription...")
whisper_model = whisper.load_model("turbo")

# Load YOLOv8n-face model
print("Loading YOLOv8n-face model for face detection...")
face_model = YOLO('.models/yolov8n-face.pt')


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


def detect_faces_in_scene(scene_path, video_id, scene_number):
    """
    Detect faces in a scene using keyframes and PyAV for faster processing

    Args:
        scene_path: Path to the scene video file
        video_id: YouTube video ID
        scene_number: Scene number (e.g., "001")

    Returns:
        Dictionary with face detection metadata
    """
    # Create directories for this video's faces
    video_faces_dir = os.path.join(
        FACES_DIR, video_id, f"Scene-{scene_number}")
    os.makedirs(video_faces_dir, exist_ok=True)

    # Path for face metadata JSON
    face_metadata_path = os.path.join(
        FACE_METADATA_DIR, f"{video_id}-Scene-{scene_number}-faces.json")

    # Check if face metadata already exists
    if os.path.exists(face_metadata_path):
        print(
            f"Face metadata already exists for {video_id} scene {scene_number}")
        with open(face_metadata_path, 'r') as f:
            face_data = json.load(f)

        # Verify that the face files actually exist
        all_faces_exist = True
        for face in face_data.get("faces", []):
            if "face_path" in face and not os.path.exists(face["face_path"]):
                all_faces_exist = False
                break

        if all_faces_exist:
            return face_data
        else:
            print(
                f"Some face images are missing for {video_id} scene {scene_number}, reprocessing...")

    print(
        f"Detecting faces in {video_id} scene {scene_number} using PyAV and keyframes...")

    face_data = {
        "video_id": video_id,
        "scene_number": scene_number,
        "scene_path": scene_path,
        "faces": []
    }

    face_count = 0
    start_time = time.time()

    try:
        # Open the video file with PyAV
        container = av.open(scene_path)

        # Get video stream
        video_stream = next(s for s in container.streams if s.type == 'video')

        # Set PyAV to skip non-key frames
        video_stream.codec_context.skip_frame = 'NONKEY'

        # Get video properties
        fps = float(video_stream.average_rate)
        duration = float(container.duration) / \
            1000000.0  # Convert from microseconds

        # Update metadata with video properties
        face_data.update({
            "fps": fps,
            "duration": duration,
            "total_frames": video_stream.frames,
        })

        # Process frames
        for frame in container.decode(video_stream):
            # Calculate timestamp in seconds
            timestamp = float(frame.pts * frame.time_base)

            # Convert PyAV frame to numpy array for YOLO
            img = frame.to_ndarray(format='rgb24')

            # Run face detection
            results = face_model(img, conf=0.25)

            # Process detected faces
            for i, detection in enumerate(results[0].boxes.data.tolist()):
                x1, y1, x2, y2, confidence, _ = detection

                # Convert to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Extract face image
                face_img = img[y1:y2, x1:x2]

                # Skip if face is too small
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue

                # Create unique face ID
                face_id = f"{video_id}_scene{scene_number}_time{timestamp:.2f}_face{i}"

                # Convert RGB to BGR for OpenCV
                face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                # Save face image
                face_filename = f"{face_id}.jpg"
                face_path = os.path.join(video_faces_dir, face_filename)
                cv2.imwrite(face_path, face_img_bgr)

                # Add face metadata
                face_data["faces"].append({
                    "face_id": face_id,
                    "timestamp": timestamp,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "face_path": face_path
                })

                face_count += 1

    except Exception as e:
        print(f"Error detecting faces in {scene_path}: {e}")
        face_data["error"] = str(e)

    finally:
        # Calculate processing time
        processing_time = time.time() - start_time

        # Add summary information
        face_data["total_faces_detected"] = face_count
        face_data["processing_time_seconds"] = processing_time

        # Save face metadata to JSON
        with open(face_metadata_path, 'w') as f:
            json.dump(face_data, f, indent=2)

    print(
        f"Detected {face_count} faces in {video_id} scene {scene_number} in {processing_time:.2f} seconds")
    return face_data


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

    # Extract scene number from the scene filename
    scene_filename = os.path.basename(scene_path)
    # Extract the Scene-XXX part from the filename
    scene_number = scene_filename.split('-Scene-')[1].split('.')[0]

    # Define output path for transcript using the same numbering as the scene file
    transcript_filename = f"{video_id}-Scene-{scene_number}.json"
    transcript_path = os.path.join(video_transcript_dir, transcript_filename)

    # Check if transcript already exists
    if os.path.exists(transcript_path):
        print(f"Transcript already exists for {video_id} scene {scene_number}")
        with open(transcript_path, 'r') as f:
            return json.load(f)

    try:
        print(f"Transcribing {video_id} scene {scene_number}...")

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


def create_consolidated_scene_data(video_id, scene_path, scene_number, face_data, transcript_data, language):
    """
    Create a consolidated JSON file for a scene that includes face metadata and transcript data

    Args:
        video_id: YouTube video ID
        scene_path: Path to the scene video file
        scene_number: Scene number (e.g., "001")
        face_data: Dictionary with face detection metadata
        transcript_data: Dictionary with transcript data

    Returns:
        Path to the consolidated JSON file
    """
    # Create the consolidated data structure
    consolidated_data = {
        # Scene-level information from face_data
        "video_id": video_id,
        "scene_number": scene_number,
        "scene_path": scene_path,
        "fps": face_data.get("fps"),
        "duration": face_data.get("duration"),
        "total_frames": face_data.get("total_frames"),

        # Include face metadata
        "face_metadata": {
            "total_faces_detected": face_data.get("total_faces_detected", 0),
            "processing_time_seconds": face_data.get("processing_time_seconds"),
            "faces": face_data.get("faces", [])
        },

        # Include transcript data
        "transcript": transcript_data,

        # Include language
        "language": language
    }

    # Create the output directory if it doesn't exist
    scene_dir = os.path.join(SCENES_DIR, video_id)
    os.makedirs(scene_dir, exist_ok=True)

    # Define the output path
    json_filename = f"{video_id}-Scene-{scene_number}.json"
    json_path = os.path.join(scene_dir, json_filename)

    # Save the consolidated data
    with open(json_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    return json_path


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

        # Check if language was previously detected in existing metadata
        detected_language = None
        for entry in existing_metadata:
            if entry['video_id'] == video_id and entry.get('language'):
                detected_language = entry['language']
                print(
                    f"Using previously detected language for {video_id}: {detected_language}")
                break

        # Only detect language if not found in existing metadata
        if detected_language is None:
            detected_language = detect_language(video_path)
            print(
                f"Video {video_id} language detected as: {detected_language}")

        # Split into scenes
        try:
            scenes = detect_and_split_scenes(video_path, video_id)
            print(f"Detected {len(scenes)} scenes for {video_id}.")
        except subprocess.CalledProcessError as e:
            print(f"Scene detection failed for {video_path}: {e}")
            return []

        print(f"Processing scenes for {video_id}")

        # Store metadata
        for scene_idx, scene_path in enumerate(scenes):
            # Extract scene number from the scene filename
            scene_filename = os.path.basename(scene_path)
            # Extract the Scene-XXX part from the filename
            scene_number = scene_filename.split('-Scene-')[1].split('.')[0]

            # Transcribe the scene
            transcript = transcribe_scene(
                scene_path, video_id, scene_idx, detected_language)

            # Detect faces in the scene
            face_data = detect_faces_in_scene(
                scene_path, video_id, scene_number)

            # Create consolidated scene data JSON
            consolidated_json_path = create_consolidated_scene_data(
                video_id, scene_path, scene_number, face_data, transcript, detected_language)

            print(f"Created consolidated scene data: {consolidated_json_path}")

            metadata_entry = {
                'video_id': video_id,
                'original_url': f"https://www.youtube.com/watch?v={video_id}",
                'video_path': video_path,
                'scene_index': scene_idx,
                'scene_number': scene_number,
                'scene_path': scene_path,
                'consolidated_data_path': consolidated_json_path,
                'language': detected_language,
            }

            # Add transcript information
            if transcript:
                transcript_filename = f"{video_id}-Scene-{scene_number}.json"
                transcript_path = os.path.join(
                    TRANSCRIPTS_DIR, video_id, transcript_filename)
                metadata_entry['transcript_path'] = transcript_path
                metadata_entry['transcript_text'] = transcript.get('text', '')
            else:
                metadata_entry['transcript_path'] = None
                metadata_entry['transcript_text'] = ''

            # Add face detection information
            face_metadata_path = os.path.join(
                FACE_METADATA_DIR, f"{video_id}-Scene-{scene_number}-faces.json")
            metadata_entry['face_metadata_path'] = face_metadata_path
            metadata_entry['face_count'] = face_data.get(
                'total_faces_detected', 0)

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
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
            existing_metadata = []

    # Process all videos - individual functions will skip work that's already done
    videos_to_process = video_ids
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

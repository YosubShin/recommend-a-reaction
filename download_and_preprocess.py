from lightasd.LightASD import setup as light_asd_setup
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
import torch
import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
import python_speech_features
import easyocr  # Add EasyOCR import

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Download, split YouTube videos into scenes, and transcribe them')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Output directory for all files')
parser.add_argument('--workers', type=int, default=1,
                    help='Number of parallel workers for processing videos')
parser.add_argument('--face_detection_fps', type=int, default=5,
                    help='FPS for face detection (higher values are more accurate but slower)')
args = parser.parse_args()

# Configuration
DEVICE = 'cpu'
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
# Directory for consolidated scene data
CONSOLIDATED_SCENE_DATA_DIR = os.path.join(OUTPUT_DIR, 'scenes')
# Directory for ASD data
ASD_DIR = os.path.join(OUTPUT_DIR, 'asd')
# Directory for extracted frames
FRAMES_DIR = os.path.join(OUTPUT_DIR, 'frames')
# Number of parallel workers
NUM_WORKERS = args.workers

# ASD parameters - hardcoded global variables instead of command line arguments
FACE_DETECTION_FPS = args.face_detection_fps
DATA_LOADER_THREAD = 10
FACE_DETECTION_SCALE = 0.25
MIN_TRACK_LENGTH = 3  # Number of min frames for each shot
NUM_FAILED_DET = 3  # Number of missed detections allowed before tracking is stopped
MIN_FACE_SIZE = 1  # Minimum face size in pixels
CROP_SCALE = 0.40  # Scale bounding box

# Thread-safe lock for metadata updates
metadata_lock = threading.Lock()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)
os.makedirs(SCENE_INFO_DIR, exist_ok=True)  # Create scene info directory
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)  # Create transcripts directory
os.makedirs(FACES_DIR, exist_ok=True)  # Create faces directory
os.makedirs(ASD_DIR, exist_ok=True)  # Create ASD directory
os.makedirs(FRAMES_DIR, exist_ok=True)  # Create frames directory

# Load whisper turbo model
print("Loading Whisper turbo model for transcription...")
whisper_model = whisper.load_model("turbo")

# Load YOLOv8n-face model
print("Loading YOLOv8n-face model for face detection...")
face_model = YOLO('.models/yolov8n-face.pt')

# Load ASD model (Light-ASD)
print("Loading Light-ASD model for active speaker detection...")
light_asd = light_asd_setup(device=DEVICE)

# Initialize EasyOCR reader (moved to global scope to avoid reinitializing for each frame)
print("Loading EasyOCR model for text detection...")
# You can add more languages as needed
ocr_reader = easyocr.Reader(['en', 'ko'])

# ASD helper functions


def convert_numpy_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    """Calculate intersection over union between two bounding boxes"""
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(sceneFaces):
    """Track faces across frames in a scene"""
    # CPU: Face tracking
    iouThres = 0.5     # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= NUM_FAILED_DET:
                    iou = bb_intersection_over_union(
                        face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > MIN_TRACK_LENGTH:
            frameNum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frameI = np.arange(frameNum[0], frameNum[-1]+1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:, 2]-bboxesI[:, 0]), np.mean(bboxesI[:, 3]-bboxesI[:, 1])) > MIN_FACE_SIZE:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video_from_frames(track, cropFile, frames, audio_file_path, start_frame=0):
    """Crop face tracks from video frames and extract corresponding audio"""
    # CPU: crop the face clips
    os.makedirs(os.path.dirname(cropFile), exist_ok=True)

    vOut = cv2.VideoWriter(
        cropFile + 't.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))  # Write video
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:  # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2)
        dets['y'].append((det[1]+det[3])/2)  # crop center x
        dets['x'].append((det[0]+det[2])/2)  # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    for fidx, frame_idx in enumerate(track['frame']):
        cs = CROP_SCALE  # Crop scale
        bs = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        # Get the frame
        if frame_idx < len(frames):
            image = frames[int(frame_idx)]
        else:
            print(
                f"Warning: Frame index {frame_idx} out of bounds ({len(frames)} frames available)")
            continue

        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                       'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X

        # Ensure coordinates are valid
        y_start = max(0, int(my-bs))
        y_end = min(frame.shape[0], int(my+bs*(1+2*cs)))
        x_start = max(0, int(mx-bs*(1+cs)))
        x_end = min(frame.shape[1], int(mx+bs*(1+cs)))

        if y_end <= y_start or x_end <= x_start:
            print(
                f"Warning: Invalid crop dimensions: {y_start}:{y_end}, {x_start}:{x_end}")
            continue

        face = frame[y_start:y_end, x_start:x_end]

        # Ensure face is not empty
        if face.size == 0:
            print(f"Warning: Empty face crop at frame {frame_idx}")
            continue

        try:
            resized_face = cv2.resize(face, (224, 224))
            vOut.write(resized_face)
        except Exception as e:
            print(f"Error resizing/writing face: {e}")
            continue

    vOut.release()

    # Extract audio segment corresponding to the face track
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1]+1) / 25

    command = (f"ffmpeg -y -i {audio_file_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "
               f"-threads {DATA_LOADER_THREAD} -ss {audioStart:.3f} -to {audioEnd:.3f} {audioTmp} -loglevel panic")

    try:
        output = subprocess.call(
            command, shell=True, stdout=None)  # Crop audio file

        # Combine audio and video
        command = (f"ffmpeg -y -i {cropFile}t.mp4 -i {audioTmp} -threads {DATA_LOADER_THREAD} "
                   f"-c:v copy -c:a aac {cropFile}.mp4 -loglevel panic")
        output = subprocess.call(command, shell=True, stdout=None)

        # Clean up temporary files
        if os.path.exists(cropFile + 't.mp4'):
            os.remove(cropFile + 't.mp4')

        return convert_numpy_to_lists({'track': track, 'proc_track': dets, 'crop_file': cropFile + '.mp4', 'audio_file': audioTmp})
    except Exception as e:
        print(f"Error processing audio/video for face track: {e}")
        return None


def evaluate_light_asd_simplified(audioFeature, videoFeature, device):
    """Run active speaker detection on a single face track"""
    # Calculate usable length
    length = min(
        (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)

    # Prepare features
    audioFeature = audioFeature[:int(round(length * 100)), :]
    videoFeature = videoFeature[:int(round(length * 25)), :, :]

    # Process entire sequence at once
    with torch.no_grad():
        inputA = torch.FloatTensor(audioFeature).unsqueeze(0).to(device)
        inputV = torch.FloatTensor(videoFeature).unsqueeze(0).to(device)

        # Forward pass
        audioEmbed = light_asd.model.forward_audio_frontend(inputA)
        visualEmbed = light_asd.model.forward_visual_frontend(inputV)
        outsAV = light_asd.model.forward_audio_visual_backend(
            audioEmbed, visualEmbed)

        # Get prediction scores
        scores = light_asd.lossAV.forward(outsAV, labels=None)

    # Round scores to one decimal place
    scores = np.round(np.array(scores), 1).astype(float)
    return scores


def evaluate_network(files, device):
    """Run active speaker detection on multiple face tracks"""
    # GPU: active speaker detection by pretrained model
    allScores = []
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split(
            '/')[-1])[0]  # Load audio and video
        audio_file = os.path.join(os.path.dirname(file), fileName + '.wav')

        try:
            _, audio = wavfile.read(audio_file)
            audioFeature = python_speech_features.mfcc(
                audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

            video = cv2.VideoCapture(file)
            videoFeature = []
            while video.isOpened():
                ret, frames = video.read()
                if ret == True:
                    face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (224, 224))
                    face = face[int(112-(112/2)):int(112+(112/2)),
                                int(112-(112/2)):int(112+(112/2))]
                    videoFeature.append(face)
                else:
                    break
            video.release()

            if len(videoFeature) == 0:
                print(f"Warning: No frames extracted from {file}")
                allScores.append(np.array([]))
                continue

            videoFeature = np.array(videoFeature)

            # Run ASD
            allScore = evaluate_light_asd_simplified(
                audioFeature, videoFeature, device)
            allScores.append(allScore)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            allScores.append(np.array([]))

    return allScores


def detect_faces_in_scene(scene_path, video_id, scene_number, fps=FACE_DETECTION_FPS):
    """
    Detect faces in a scene at specified FPS

    Args:
        scene_path: Path to the scene video file
        video_id: YouTube video ID
        scene_number: Scene number (e.g., "001")
        fps: Target FPS for face detection (default: FACE_DETECTION_FPS)

    Returns:
        Dictionary with face detection metadata
    """
    # Create directories for this video's faces
    video_faces_dir = os.path.join(
        FACES_DIR, video_id, f"Scene-{scene_number}")
    os.makedirs(video_faces_dir, exist_ok=True)

    # Path for face metadata JSON
    face_metadata_path = os.path.join(
        FACES_DIR, video_id, f"Scene-{scene_number}.json")

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
        f"Detecting faces in {video_id} scene {scene_number} at {fps} FPS...")

    face_data = {
        "video_id": video_id,
        "scene_number": scene_number,
        "scene_path": scene_path,
        "faces": [],
    }

    face_count = 0
    start_time = time.time()

    try:
        # Open the video file with PyAV
        container = av.open(scene_path)

        # Get video stream
        video_stream = next(s for s in container.streams if s.type == 'video')

        # Get video properties
        video_fps = float(video_stream.average_rate)
        duration = float(container.duration) / \
            1000000.0  # Convert from microseconds
        total_frames = video_stream.frames

        # Update metadata with video properties
        face_data.update({
            "fps": video_fps,
            "duration": duration,
            "total_frames": total_frames,
        })

        # Calculate frame sampling rate
        sample_rate = max(1, int(video_fps / fps))
        face_data["face_detection_fps"] = fps
        face_data["sample_rate"] = sample_rate

        # Process frames
        frame_idx = 0
        frame_faces_list = []

        for frame in container.decode(video_stream):
            # Skip frames based on sample rate
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            # Calculate timestamp in seconds
            timestamp = float(frame.pts * frame.time_base)

            # Convert PyAV frame to numpy array for YOLO
            img = frame.to_ndarray(format='rgb24')

            # Run face detection
            results = face_model(img, conf=0.25, verbose=False)

            frame_faces = []
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

                # Create unique face ID for filename
                face_filename = f"{video_id}_scene{scene_number}_time{timestamp:.2f}_face{i}.jpg"
                face_path = os.path.join(video_faces_dir, face_filename)

                # Convert RGB to BGR for OpenCV
                face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                # Save face image
                cv2.imwrite(face_path, face_img_bgr)

                # Add face metadata
                frame_faces.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "bbox": [x1, y1, x2, y2],
                    "conf": confidence,
                    "face_path": face_path
                })

                face_count += 1

            frame_faces_list.append(frame_faces)
            frame_idx += 1
    except Exception as e:
        print(f"Error detecting faces in {scene_path}: {e}")
        face_data["error"] = str(e)

    finally:
        # Calculate processing time
        processing_time = time.time() - start_time

        # Add summary information
        face_data["total_faces_detected"] = face_count
        face_data["processing_time_seconds"] = processing_time
        face_data["faces"] = frame_faces_list

        # Save face metadata to JSON
        with open(face_metadata_path, 'w') as f:
            json.dump(face_data, f, indent=2)

    print(
        f"Detected {face_count} faces in {video_id} scene {scene_number} in {processing_time:.2f} seconds")
    return face_data


def process_asd(scene_path, video_id, scene_number, face_detection_results, device=DEVICE):
    """
    Process active speaker detection for a scene

    Args:
        scene_path: Path to the scene video file
        video_id: YouTube video ID
        scene_number: Scene number (e.g., "001")
        light_asd: TalkNet model instance

    Returns:
        Dictionary with ASD results
    """
    print(f"Processing ASD for {video_id} scene {scene_number}...")

    # Create output directories
    scene_asd_dir = os.path.join(ASD_DIR, video_id, f"Scene-{scene_number}")
    scene_crops_dir = os.path.join(
        scene_asd_dir, "crops")
    os.makedirs(scene_asd_dir, exist_ok=True)
    os.makedirs(scene_crops_dir, exist_ok=True)

    # Path for ASD results
    asd_results_path = os.path.join(
        ASD_DIR, video_id, f"Scene-{scene_number}.json")

    # Check if ASD results already exist
    if os.path.exists(asd_results_path):
        print(f"ASD results already exist for {video_id} scene {scene_number}")
        with open(asd_results_path, 'r') as f:
            return json.load(f)

    # Step 1: Get face detection results
    if not face_detection_results:
        print(f"Face detection failed for {video_id} scene {scene_number}")
        return None

    # Step 2: Load all frames for face tracking and cropping
    cap = cv2.VideoCapture(scene_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {scene_path}")
        return None

    # Load all frames if the video is short enough
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Step 3: Get face detections from the results
    face_detections = face_detection_results["faces"]

    # Step 4: Track faces across frames
    print(f"Tracking faces in {video_id} scene {scene_number}...")
    face_tracks = track_shot(face_detections)
    print(f"Found {len(face_tracks)} face tracks")

    # Step 5: Crop face tracks and extract audio
    print(f"Cropping face tracks in {video_id} scene {scene_number}...")
    video_tracks = []
    for i, track in enumerate(face_tracks):
        crop_file = os.path.join(scene_crops_dir, f"{i:05d}")
        track_result = crop_video_from_frames(
            track, crop_file, frames, scene_path)
        if track_result:
            video_tracks.append(track_result)

    # Step 6: Run ASD on face tracks
    print(f"Running ASD on {len(video_tracks)} face tracks...")
    crop_files = [track["crop_file"]
                  for track in video_tracks if "crop_file" in track]
    asd_scores = evaluate_network(crop_files, device)

    # Add ASD scores to track results
    for i, (track, scores) in enumerate(zip(video_tracks, asd_scores)):
        if len(scores) > 0:
            track["asd_scores"] = scores.tolist()

    # Save ASD results
    asd_results = {
        "video_id": video_id,
        "scene_number": scene_number,
        "scene_path": scene_path,
        "face_tracks": len(video_tracks),
        "tracks": video_tracks
    }

    with open(asd_results_path, 'w') as f:
        json.dump(asd_results, f, indent=2)

    return asd_results


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
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

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


def extract_frames(scene_path, video_id, scene_number):
    """
    Extract keyframes from a scene and create masked versions with text removed

    Args:
        scene_path: Path to the scene video file
        video_id: YouTube video ID
        scene_number: Scene number (e.g., "001")

    Returns:
        Dictionary with frame extraction metadata
    """
    # Create directories for this video's frames
    video_frames_dir = os.path.join(
        FRAMES_DIR, video_id, f"Scene-{scene_number}")
    os.makedirs(video_frames_dir, exist_ok=True)

    # Path for frame metadata JSON
    frame_metadata_path = os.path.join(
        FRAMES_DIR, video_id, f"Scene-{scene_number}.json")

    # Check if frame metadata already exists
    if os.path.exists(frame_metadata_path):
        print(
            f"Frame metadata already exists for {video_id} scene {scene_number}")
        with open(frame_metadata_path, 'r') as f:
            frame_data = json.load(f)

        # Verify that the frame files actually exist
        all_frames_exist = True
        for frame in frame_data.get("frames", []):
            if "frame_path" in frame and not os.path.exists(frame["frame_path"]):
                all_frames_exist = False
                break
            # Also check if masked frames exist
            if "masked_frame_path" in frame and not os.path.exists(frame["masked_frame_path"]):
                all_frames_exist = False
                break

        if all_frames_exist:
            return frame_data
        else:
            print(
                f"Some frame images are missing for {video_id} scene {scene_number}, reprocessing...")

    print(
        f"Extracting keyframes from {video_id} scene {scene_number}...")

    frame_data = {
        "video_id": video_id,
        "scene_number": scene_number,
        "scene_path": scene_path,
        "frames": [],
    }

    frame_count = 0
    start_time = time.time()

    try:
        # Open the video file with PyAV
        container = av.open(scene_path)

        # Get video stream
        video_stream = next(s for s in container.streams if s.type == 'video')

        # Configure to only decode keyframes
        video_stream.thread_type = 'AUTO'
        # Only decode keyframes
        video_stream.codec_context.skip_frame = 'NONKEY'

        # Get video properties
        video_fps = float(video_stream.average_rate)
        duration = float(container.duration) / \
            1000000.0  # Convert from microseconds
        total_frames = video_stream.frames

        # Update metadata with video properties
        frame_data.update({
            "fps": video_fps,
            "duration": duration,
            "total_frames": total_frames,
            "extraction_mode": "keyframes_only"
        })

        # Process frames
        extracted_frames = []

        for frame in container.decode(video_stream):
            # Calculate timestamp in seconds
            timestamp = float(frame.pts * frame.time_base)

            # Convert PyAV frame to numpy array
            img = frame.to_ndarray(format='rgb24')

            # Create unique frame filename
            frame_filename = f"{video_id}_scene{scene_number}_time{timestamp:.2f}.jpg"
            masked_frame_filename = f"{video_id}_scene{scene_number}_time{timestamp:.2f}_masked.jpg"
            frame_path = os.path.join(video_frames_dir, frame_filename)
            masked_frame_path = os.path.join(
                video_frames_dir, masked_frame_filename)

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Save original frame image
            cv2.imwrite(frame_path, img_bgr)

            # Create masked version with text removed
            try:
                # Detect text using EasyOCR
                result = ocr_reader.readtext(img_bgr)

                # Create a copy of the image for masking
                masked_img = img_bgr.copy()

                # Track if any text was detected and masked
                text_detected = False
                detected_text = []

                # Loop through each detected text region
                for detection in result:
                    # Get bounding box coordinates and text
                    bbox = detection[0]
                    text = detection[1]
                    confidence = detection[2]

                    # Convert to integer points for OpenCV
                    points = np.array([[int(p[0]), int(p[1])] for p in bbox])

                    # Create a mask for this region
                    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)

                    # Apply a black rectangle to mask the text
                    masked_img[mask > 0] = [0, 0, 0]  # Black color

                    text_detected = True
                    detected_text.append(
                        {"text": text, "confidence": confidence})

                # Save the masked image
                cv2.imwrite(masked_frame_path, masked_img)

            except Exception as e:
                print(f"Error masking text in frame: {e}")
                # If text masking fails, just copy the original frame
                cv2.imwrite(masked_frame_path, img_bgr)
                text_detected = False
                detected_text = []

            # Add frame metadata
            frame_info = {
                "frame_index": int(timestamp * video_fps),
                "timestamp": timestamp,
                "is_keyframe": True,  # All frames we get are keyframes
                "frame_path": frame_path,
                "masked_frame_path": masked_frame_path,
                "text_detected": text_detected,
                "detected_text": detected_text
            }

            extracted_frames.append(frame_info)
            frame_count += 1

    except Exception as e:
        print(f"Error extracting frames from {scene_path}: {e}")
        frame_data["error"] = str(e)

    finally:
        # Calculate processing time
        processing_time = time.time() - start_time

        # Add summary information
        frame_data["total_frames_extracted"] = frame_count
        frame_data["processing_time_seconds"] = processing_time
        frame_data["frames"] = extracted_frames

        # Save frame metadata to JSON
        with open(frame_metadata_path, 'w') as f:
            json.dump(frame_data, f, indent=2)

    print(
        f"Extracted {frame_count} keyframes from {video_id} scene {scene_number} in {processing_time:.2f} seconds")
    return frame_data


def create_consolidated_scene_data(video_id, scene_path, scene_number, face_data, transcript_data, language, asd_data=None, frame_data=None):
    """
    Create a consolidated JSON file for a scene that includes face metadata,

    Args:
        video_id: YouTube video ID
        scene_path: Path to the scene video file
        scene_number: Scene number (e.g., "001")
        face_data: Dictionary with face detection metadata
        transcript_data: Dictionary with transcript data
        language: Detected language for the scene
        asd_data: Dictionary with ASD results
        frame_data: Dictionary with extracted frames metadata

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

    # Add ASD data if it exists
    if asd_data:
        consolidated_data["asd_data"] = asd_data

    # Add frame data if it exists
    if frame_data:
        consolidated_data["frame_metadata"] = {
            "total_frames_extracted": frame_data.get("total_frames_extracted", 0),
            "extraction_mode": frame_data.get("extraction_mode", "keyframes_only"),
            "processing_time_seconds": frame_data.get("processing_time_seconds"),
            "frames": frame_data.get("frames", [])
        }

    # Create the output directory if it doesn't exist
    scene_dir = os.path.join(SCENES_DIR, video_id)
    os.makedirs(scene_dir, exist_ok=True)

    # Define the output path
    json_filename = f"{video_id}-Scene-{scene_number}.json"
    json_path = os.path.join(scene_dir, json_filename)

    # Save the consolidated data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)

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

            # Process ASD for the scene
            asd_data = process_asd(scene_path, video_id,
                                   scene_number, face_data)

            # Extract frames from the scene
            frame_data = extract_frames(scene_path, video_id, scene_number)

            # Create consolidated scene data JSON
            consolidated_json_path = create_consolidated_scene_data(
                video_id, scene_path, scene_number, face_data, transcript,
                detected_language, asd_data, frame_data)

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
                FACES_DIR, video_id, f"Scene-{scene_number}.json")
            metadata_entry['face_metadata_path'] = face_metadata_path
            metadata_entry['face_count'] = face_data.get(
                'total_faces_detected', 0)

            # Add ASD information
            if asd_data:
                asd_results_path = os.path.join(
                    ASD_DIR, video_id, f"Scene-{scene_number}.json")
                metadata_entry['asd_results_path'] = asd_results_path
                metadata_entry['asd_face_tracks'] = asd_data.get(
                    'face_tracks', 0)

            # Add frame extraction information
            frame_metadata_path = os.path.join(
                FRAMES_DIR, video_id, f"Scene-{scene_number}.json")
            metadata_entry['frame_metadata_path'] = frame_metadata_path
            metadata_entry['frame_count'] = frame_data.get(
                'total_frames_extracted', 0)

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

import json
from deepface import DeepFace
import cv2
import os


def load_scene_data(scene_path):
    with open(scene_path, 'r') as file:
        return json.load(file)


def is_context_reaction_pair(context_scene, reaction_scene):
    """
    Determine if two consecutive scenes form a context-reaction shot pair.

    Args:
        context_scene (dict): The first scene JSON data
        reaction_scene (dict): The second scene JSON data

    Returns:
        bool: True if the scenes form a context-reaction pair, False otherwise
    """
    # Get scene identifiers for logging
    context_id = context_scene.get("video_id", "unknown")
    reaction_id = reaction_scene.get("video_id", "unknown")
    context_scene_num = context_scene.get("scene_number", "unknown")
    reaction_scene_num = reaction_scene.get("scene_number", "unknown")
    scene_pair_id = f"{context_id}:{context_scene_num}-{reaction_scene_num}"

    print(f"\nEvaluating potential context-reaction pair: {scene_pair_id}")

    # Check if context shot duration is between 1-6 seconds
    context_duration = context_scene.get("duration", 0)
    if context_duration < 1 or context_duration > 6:
        print(
            f"FAIL: Context duration ({context_duration}s) not between 1-6 seconds")
        return False

    # Check if reaction shot is less than 3 seconds
    reaction_duration = reaction_scene.get("duration", 0)
    if reaction_duration > 3:
        print(
            f"FAIL: Reaction duration ({reaction_duration}s) exceeds 3 seconds")
        return False

    # Check if both shots have faces
    context_faces = context_scene.get(
        "face_metadata", {}).get("total_faces_detected", 0)
    reaction_faces = reaction_scene.get(
        "face_metadata", {}).get("total_faces_detected", 0)
    if context_faces == 0 or reaction_faces == 0:
        print(
            f"FAIL: Missing faces in shots (context: {context_faces}, reaction: {reaction_faces})")
        return False

    # Check if there's speech in the context shot
    context_transcript = context_scene.get(
        "transcript", {}).get("text", "").strip()
    if not context_transcript:
        print(f"FAIL: No speech transcript in context shot")
        return False

    # Check if there is an active speaker in the context shot
    has_active_speaker = False
    asd_data = context_scene.get("asd_data", {})
    tracks = asd_data.get("tracks", [])
    for track in tracks:
        asd_scores = track.get("asd_scores", [])
        if any(score > 0 for score in asd_scores):
            has_active_speaker = True
            break

    if not has_active_speaker:
        print(f"FAIL: No active speaker detected in context shot")
        return False

    # Check if there is an active speaker in the reaction shot (should NOT have one)
    reaction_has_active_speaker = False
    reaction_asd_data = reaction_scene.get("asd_data", {})
    reaction_tracks = reaction_asd_data.get("tracks", [])
    for track in reaction_tracks:
        asd_scores = track.get("asd_scores", [])
        if any(score > 0 for score in asd_scores):
            reaction_has_active_speaker = True
            break

    if reaction_has_active_speaker:
        print(
            f"FAIL: Active speaker detected in reaction shot (reactions should be silent)")
        return False

    # Check if it's an L-cut by comparing faces using DeepFace
    # Get face images from the last frame of context scene
    context_face_images = []
    context_faces = context_scene.get("face_metadata", {}).get("faces", [])
    if context_faces:
        # Get faces from the last frame
        last_frame_faces = context_faces[-1]

        for face in last_frame_faces:
            face_image_path = face.get("face_path", "")
            if os.path.exists(face_image_path):
                context_face_images.append(cv2.imread(face_image_path))
            else:
                print(f"Warning: Face image not found at {face_image_path}")

    # Get face images from the first frame of reaction scene
    reaction_face_images = []
    reaction_faces = reaction_scene.get("face_metadata", {}).get("faces", [])
    if reaction_faces:
        # Get faces from the first frame
        first_frame_faces = reaction_faces[0]

        for face in first_frame_faces:
            face_image_path = face.get("face_path", "")
            if os.path.exists(face_image_path):
                reaction_face_images.append(cv2.imread(face_image_path))
            else:
                print(f"Warning: Face image not found at {face_image_path}")

    # Compare each context face with each reaction face
    face_match_found = False
    for i, context_face in enumerate(context_face_images):
        for j, reaction_face in enumerate(reaction_face_images):
            try:
                result = DeepFace.verify(
                    context_face, reaction_face, model_name="ArcFace")
                if result["verified"]:
                    # Same person found in both shots, not an L-cut
                    print(
                        f"FAIL: Same person found in both shots (context face {i+1}, reaction face {j+1})")
                    face_match_found = True
                    break
            except Exception as e:
                # If comparison fails, continue with other faces
                print(f"Face comparison error: {e}")
                continue
        if face_match_found:
            break

    if face_match_found:
        return False

    # If all criteria are met, it's a context-reaction pair
    print(
        f"SUCCESS: All criteria met for context-reaction pair {scene_pair_id}")
    return True


def test_context_reaction_pair():
    # positive example
    context_path = "/Users/yosub/co/recommend-a-reaction/output/scenes/jajw-8Bj_Ys/jajw-8Bj_Ys-Scene-017.json"
    reaction_path = "/Users/yosub/co/recommend-a-reaction/output/scenes/jajw-8Bj_Ys/jajw-8Bj_Ys-Scene-018.json"

    context_scene = load_scene_data(context_path)
    reaction_scene = load_scene_data(reaction_path)

    print(is_context_reaction_pair(context_scene, reaction_scene))

    # negative example - no active speaker in the context shot
    context_path = "/Users/yosub/co/recommend-a-reaction/output/scenes/jajw-8Bj_Ys/jajw-8Bj_Ys-Scene-020.json"
    reaction_path = "/Users/yosub/co/recommend-a-reaction/output/scenes/jajw-8Bj_Ys/jajw-8Bj_Ys-Scene-021.json"

    context_scene = load_scene_data(context_path)
    reaction_scene = load_scene_data(reaction_path)

    print(is_context_reaction_pair(context_scene, reaction_scene))

    # positive example
    context_path = "/Users/yosub/co/recommend-a-reaction/output/scenes/jajw-8Bj_Ys/jajw-8Bj_Ys-Scene-540.json"
    reaction_path = "/Users/yosub/co/recommend-a-reaction/output/scenes/jajw-8Bj_Ys/jajw-8Bj_Ys-Scene-541.json"

    context_scene = load_scene_data(context_path)
    reaction_scene = load_scene_data(reaction_path)

    print(is_context_reaction_pair(context_scene, reaction_scene))


def find_context_reaction_pairs(output_dir):
    """
    Find all context-reaction shot pairs in the given output directory.

    Args:
        output_dir (str): Path to the output directory containing scene data

    Returns:
        list: List of dictionaries containing context-reaction pair information
    """
    scenes_dir = os.path.join(output_dir, "scenes")
    results = []

    # Check if scenes directory exists
    if not os.path.exists(scenes_dir):
        print(f"Error: Scenes directory not found at {scenes_dir}")
        return results

    # Get all video directories
    video_dirs = [d for d in os.listdir(
        scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))]

    for video_id in video_dirs:
        video_dir = os.path.join(scenes_dir, video_id)
        scene_files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith('.json')])

        print(f"Processing video {video_id} with {len(scene_files)} scenes")

        # Process consecutive scene pairs
        for i in range(len(scene_files) - 1):
            context_path = os.path.join(video_dir, scene_files[i])
            reaction_path = os.path.join(video_dir, scene_files[i + 1])

            try:
                context_scene = load_scene_data(context_path)
                reaction_scene = load_scene_data(reaction_path)

                is_pair = is_context_reaction_pair(
                    context_scene, reaction_scene)

                # Get transcript text for context and reaction scenes
                context_transcript = context_scene.get(
                    "transcript", {}).get("text", "").strip()
                reaction_transcript = reaction_scene.get(
                    "transcript", {}).get("text", "").strip()

                # Extract emotion data for context scene
                context_emotions = extract_emotion_data(context_scene)

                # Extract emotion data for reaction scene
                reaction_emotions = extract_emotion_data(reaction_scene)

                # Create result entry
                result = {
                    "video_id": video_id,
                    "context_scene": scene_files[i],
                    "reaction_scene": scene_files[i + 1],
                    "context_scene_number": context_scene.get("scene_number", "unknown"),
                    "reaction_scene_number": reaction_scene.get("scene_number", "unknown"),
                    "is_context_reaction_pair": is_pair,
                    "context_duration": context_scene.get("duration", 0),
                    "reaction_duration": reaction_scene.get("duration", 0),
                    "context_transcript": context_transcript,
                    "reaction_transcript": reaction_transcript,
                    "context_emotions": context_emotions,
                    "reaction_emotions": reaction_emotions
                }

                # Get keyframe for context scene
                context_frames = context_scene.get(
                    "frame_metadata", {}).get("frames", [])
                if context_frames:
                    middle_idx = len(context_frames) // 2
                    context_middle_frame = context_frames[middle_idx].get(
                        "masked_frame_path", "")
                    result["context_middle_frame"] = context_middle_frame
                else:
                    result["context_middle_frame"] = ""

                # Get keyframe for reaction scene
                reaction_frames = reaction_scene.get(
                    "frame_metadata", {}).get("frames", [])
                if reaction_frames:
                    middle_idx = len(reaction_frames) // 2
                    reaction_middle_frame = reaction_frames[middle_idx].get(
                        "masked_frame_path", "")
                    result["reaction_middle_frame"] = reaction_middle_frame
                else:
                    result["reaction_middle_frame"] = ""

                results.append(result)

            except Exception as e:
                print(
                    f"Error processing scene pair {scene_files[i]} and {scene_files[i + 1]}: {e}")

    return results


def extract_emotion_data(scene):
    """
    Extract emotion data from a scene.

    Args:
        scene (dict): Scene data

    Returns:
        str: Comma-separated emotion data from the scene
    """
    emotion_data = scene.get("emotion_data", {})
    tracks = emotion_data.get("tracks", [])

    if not tracks:
        return ""

    all_emotions = []

    for track in tracks:
        faces = track.get("faces", [])
        if not faces:
            continue

        # Get the middle face in the track as representative
        middle_idx = len(faces) // 2
        middle_face = faces[middle_idx]

        emotion = middle_face.get("emotion", {})
        if emotion:
            # Format the emotion data as a string
            emotion_str = ";".join(
                [f"{k}:{v:.2f}" for k, v in emotion.items()])
            all_emotions.append(emotion_str)

    # Join all emotions with commas
    return ",".join(all_emotions)


def save_results_to_csv(results, output_dir):
    """
    Save context-reaction pair results to a CSV file.

    Args:
        results (list): List of dictionaries containing context-reaction pair information
        output_dir (str): Path to the output directory
    """
    import csv

    csv_path = os.path.join(output_dir, "context_reaction_pairs.csv")

    if not results:
        print("No results to save.")
        return

    # Get fieldnames from the first result
    fieldnames = results[0].keys()

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_path}")


def extract_context_reaction_pairs(output_dir):
    """
    Extract context-reaction shot pairs from scene data and save results to CSV.

    Args:
        output_dir (str): Path to the output directory
    """
    print(f"Extracting context-reaction pairs from {output_dir}")
    results = find_context_reaction_pairs(output_dir)

    # Count positive pairs
    positive_pairs = sum(1 for r in results if r["is_context_reaction_pair"])

    print(
        f"Found {positive_pairs} context-reaction pairs out of {len(results)} consecutive scene pairs")

    # Save results to CSV
    save_results_to_csv(results, output_dir)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        extract_context_reaction_pairs(output_dir)
    else:
        print("Usage: python extract_context_reaction_pairs.py <output_directory>")
        # For testing purposes
        # test_context_reaction_pair()

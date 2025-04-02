from insightface.app import FaceAnalysis
import os
import shutil
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import cv2
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import os.path as osp

# Define paths
input_dir = 'output/faces'
tracked_faces_dir = 'output/asd'  # Directory with tracked faces
output_dir = 'output/clustered_faces'
# Specify the video ID to process
target_video_id = "jajw-8Bj_Ys"


def get_image(image_file, to_rgb=False):

    if not osp.exists(image_file):
        raise FileNotFoundError(f'{image_file} not found')
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:, :, ::-1]
    return img


app = FaceAnalysis(name='antelopev2')  # or use 'buffalo_l' for more speed
app.prepare(ctx_id=-1, det_size=(160, 160))  # or -1 for CPU


# Remove output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def get_face_size(image_path):
    """Get the height of the face in pixels"""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    height, width = img.shape[:2]
    return height


# Collect tracked faces
track_paths = defaultdict(list)
track_keys = []

# Path to tracked faces: 'output/asd/<video_id>/Scene-<scene_number>/track_<track index>'
video_path = os.path.join(tracked_faces_dir, target_video_id)
if os.path.exists(video_path):
    for scene_dir in os.listdir(video_path):
        scene_path = os.path.join(video_path, scene_dir)
        if not os.path.isdir(scene_path) or not scene_dir.startswith('Scene-'):
            continue

        # Get face sizes from the Scene JSON file
        scene_json_path = os.path.join(video_path, f"{scene_dir}.json")
        face_sizes = []

        scene_data = None
        if os.path.exists(scene_json_path):
            try:
                with open(scene_json_path, 'r') as f:
                    scene_data = json.load(f)
            except Exception as e:
                print(f"Error reading scene JSON: {e}")

        # Find all track directories
        for item in os.listdir(scene_path):
            track_dir = os.path.join(scene_path, item)
            if os.path.isdir(track_dir) and item.startswith('track_'):
                # Get all face images in this track
                face_images = [os.path.join(track_dir, f) for f in os.listdir(track_dir)
                               if f.endswith('.jpg') or f.endswith('.png')]

                if face_images:
                    # Find the track that matches our track_idx
                    track_idx = int(os.path.basename(track_dir).split('_')[1])
                    for track in scene_data.get('tracks', []):
                        if track.get('track_idx') == track_idx:
                            # Get the face sizes from proc_track.s
                            face_sizes = track.get(
                                'proc_track', {}).get('s', [])
                            break

                    # Filter for faces that are large enough
                    if face_sizes:
                        # Use the sizes from the JSON file
                        valid_faces = []
                        for i, face_image in enumerate(face_images):
                            # 10% of 1080px
                            if i < len(face_sizes) and face_sizes[i] >= 108:
                                valid_faces.append(face_image)
                    else:
                        # Fall back to the original method if JSON data isn't available
                        valid_faces = [
                            f for f in face_images if get_face_size(f) >= 108]

                    if valid_faces:
                        track_key = f"{target_video_id}_{scene_dir}_{item}"
                        track_paths[track_key] = valid_faces
                        track_keys.append(track_key)

print(f"Found {len(track_keys)} face tracks to process")

# Extract embeddings for each track by averaging multiple face embeddings
embeddings = []
valid_track_keys = []

# Define embeddings cache file
embeddings_cache_file = f'output/embeddings_{target_video_id}.npz'

# Create a dictionary to store cached embeddings
cached_embeddings = {}

# Check if embeddings cache exists
if os.path.exists(embeddings_cache_file):
    print(f"Loading embeddings from cache: {embeddings_cache_file}")
    cache_data = np.load(embeddings_cache_file, allow_pickle=True)
    cached_embeddings_array = cache_data['embeddings']
    cached_track_keys = cache_data['track_keys'].tolist()

    # Create a dictionary mapping track keys to embeddings
    for i, track_key in enumerate(cached_track_keys):
        cached_embeddings[track_key] = cached_embeddings_array[i]

    print(f"Loaded {len(cached_embeddings)} embeddings from cache")

print("Processing embeddings for tracks...")
for track_key in tqdm(track_keys, desc="Extracting embeddings for tracks"):
    # Check if we already have this embedding in cache
    if track_key in cached_embeddings:
        embeddings.append(cached_embeddings[track_key])
        valid_track_keys.append(track_key)
        continue

    # If not in cache, calculate the embedding
    face_paths = track_paths[track_key]

    # Select up to 5 temporally distant frames
    if len(face_paths) > 5:
        # Sort by filename which should preserve temporal order
        face_paths.sort()
        # Take evenly spaced samples
        step = len(face_paths) // 5
        selected_faces = face_paths[::step][:5]
    else:
        selected_faces = face_paths

    track_embeddings = []

    # Extract embeddings for each selected face in the track
    for face_path in selected_faces:
        faces = app.get(get_image(face_path), max_num=1)
        if len(faces) == 0:
            print(f"No face detected in {face_path}")
            continue
        embedding = faces[0].normed_embedding  # 512D vector
        track_embeddings.append(embedding)

    # Only use tracks where we could extract at least five embedding
    if track_embeddings and len(track_embeddings) >= 5:
        # Average the embeddings for this track
        avg_embedding = np.mean(track_embeddings, axis=0)
        embeddings.append(avg_embedding)
        valid_track_keys.append(track_key)
        # Add to cache dictionary
        cached_embeddings[track_key] = avg_embedding

embeddings = np.array(embeddings)
print(f"Successfully processed {len(embeddings)} tracks")

# Save updated embeddings to cache
print(f"Saving embeddings to cache: {embeddings_cache_file}")
# Convert the dictionary back to arrays for saving
all_track_keys = list(cached_embeddings.keys())
all_embeddings = np.array([cached_embeddings[k] for k in all_track_keys])
np.savez(embeddings_cache_file,
         embeddings=all_embeddings,
         track_keys=np.array(all_track_keys, dtype=object))

# 2. Try Agglomerative Clustering with different distance thresholds
print("\nTrying Agglomerative Clustering...")
for threshold in [0.5]:
    agg_clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    labels = agg_clustering.fit_predict(embeddings)
    n_clusters = len(set(labels))

    # Count samples per cluster
    cluster_counts = {}
    for label in set(labels):
        cluster_counts[label] = np.sum(labels == label)

    print(f"Threshold {threshold}: {n_clusters} clusters")
    # Print the sizes of the top 5 clusters
    sorted_clusters = sorted(cluster_counts.items(),
                             key=lambda x: x[1], reverse=True)
    print(f"  Top clusters: {sorted_clusters[:5]}")

    if 5 <= n_clusters <= 30:
        print(f"  This threshold gives a reasonable number of clusters")
        agg_labels = labels

print("\nUsing Agglomerative Clustering results")
labels = agg_labels

# Visualize the chosen clustering
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
unique_labels = set(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Black used for noise
        color = [0, 0, 0, 1]

    mask = labels == label
    plt.scatter(
        reduced_embeddings[mask, 0],
        reduced_embeddings[mask, 1],
        c=[color],
        label=f"Cluster {label}" if label != -1 else "Noise"
    )

plt.title('Clustering Results (PCA Visualization)')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.savefig('clustering_results.png', bbox_inches='tight')

# Group tracks by cluster
tracks_by_cluster = defaultdict(list)
for i, label in enumerate(labels):
    tracks_by_cluster[label].append(valid_track_keys[i])

# Copy representative faces to output directory
for label, track_keys_list in tracks_by_cluster.items():
    if label == -1:
        # Skip noise
        print(f"Skipping {len(track_keys_list)} tracks classified as noise")
        continue

    # Create directory for this identity
    identity_dir = os.path.join(output_dir, f"person_{label}")
    os.makedirs(identity_dir, exist_ok=True)

    print(f"Person {label}: {len(track_keys_list)} tracks")

    # Copy representative files from each track
    for i, track_key in enumerate(track_keys_list):
        # Get all faces for this track
        face_paths = track_paths[track_key]

        # Skip tracks with no valid faces
        if len(face_paths) == 0:
            print(f"Skipping track with no valid faces: {track_key}")
            continue

        # Choose a representative face (middle of the track)
        representative_face = face_paths[len(face_paths) // 2]

        # Create a filename that preserves track information
        filename = f"{i:04d}_{track_key}_{os.path.basename(representative_face)}"
        shutil.copy(representative_face, os.path.join(identity_dir, filename))

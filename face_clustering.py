import os
import shutil
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import cv2
from tqdm import tqdm
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random

# Define paths
input_dir = 'output/faces'
tracked_faces_dir = 'output/asd'  # Directory with tracked faces
output_dir = 'output/clustered_faces'
# Specify the video ID to process
target_video_id = "jajw-8Bj_Ys"

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

        # Find all track directories
        for item in os.listdir(scene_path):
            track_dir = os.path.join(scene_path, item)
            if os.path.isdir(track_dir) and item.startswith('track_'):
                # Get all face images in this track
                face_images = [os.path.join(track_dir, f) for f in os.listdir(track_dir)
                               if f.endswith('.jpg') or f.endswith('.png')]

                if face_images:
                    # Filter for faces that are large enough
                    valid_faces = [f for f in face_images if get_face_size(
                        f) >= 108]  # 10% of 1080px

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

# Check if embeddings cache exists
if os.path.exists(embeddings_cache_file):
    print(f"Loading embeddings from cache: {embeddings_cache_file}")
    cache_data = np.load(embeddings_cache_file, allow_pickle=True)
    embeddings = cache_data['embeddings']
    valid_track_keys = cache_data['track_keys'].tolist()
    print(f"Loaded {len(embeddings)} embeddings from cache")
else:
    print("Calculating embeddings (this may take a while)...")
    for track_key in tqdm(track_keys, desc="Extracting embeddings for tracks"):
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
            try:
                embedding = DeepFace.represent(
                    face_path, model_name="ArcFace", enforce_detection=False,
                    normalization="ArcFace", align=True)[0]["embedding"]
                track_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {face_path}: {e}")

        # Only use tracks where we could extract at least one embedding
        if track_embeddings:
            # Average the embeddings for this track
            avg_embedding = np.mean(track_embeddings, axis=0)
            embeddings.append(avg_embedding)
            valid_track_keys.append(track_key)

    embeddings = np.array(embeddings)
    print(f"Successfully processed {len(embeddings)} tracks")

    # Save embeddings to cache
    print(f"Saving embeddings to cache: {embeddings_cache_file}")
    np.savez(embeddings_cache_file,
             embeddings=embeddings,
             track_keys=np.array(valid_track_keys, dtype=object))

# Try different clustering methods
print("Trying different clustering methods...")

# 1. Try KMeans with different numbers of clusters
best_score = -1
best_k = 0
best_labels = None

for k in range(10, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Calculate silhouette score to evaluate clustering quality
    if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
        score = silhouette_score(embeddings, labels)
        print(f"KMeans with {k} clusters: silhouette score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

print(
    f"\nBest KMeans clustering: {best_k} clusters with silhouette score {best_score:.4f}")

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

# Use the best clustering method
if 'agg_labels' in locals() and len(set(agg_labels)) >= 3:
    print("\nUsing Agglomerative Clustering results")
    labels = agg_labels
elif best_score > 0:
    print("\nUsing KMeans clustering results")
    labels = best_labels
else:
    print("\nFalling back to default clustering")
    # Try a much larger eps value for DBSCAN
    clustering = DBSCAN(eps=3.0, min_samples=5).fit(embeddings)
    labels = clustering.labels_

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

        # Choose a representative face (middle of the track)
        representative_face = face_paths[len(face_paths) // 2]

        # Create a filename that preserves track information
        filename = f"{i:04d}_{os.path.basename(representative_face)}"
        shutil.copy(representative_face, os.path.join(identity_dir, filename))

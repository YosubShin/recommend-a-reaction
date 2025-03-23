import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import defaultdict


def get_face_size_ratios_by_count(faces_dir="output/faces"):
    """
    Extract face sizes from JSON files and calculate the ratio of max dimension to 1080.
    Organize data by the number of faces in each frame (one-shot, two-shot, etc.)
    Only considers the first frame's faces for each scene.
    """
    # Dictionary to store face ratios grouped by number of faces in frame
    face_ratios_by_count = defaultdict(list)

    # Find all JSON files in the faces directory
    json_files = glob.glob(os.path.join(faces_dir, "*", "*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if 'faces' key exists and has data
            if 'faces' in data and data['faces'] and len(data['faces']) > 0:
                # Get faces from the first frame
                first_frame_faces = data['faces'][0]

                # Count number of faces in this frame
                face_count = len(first_frame_faces)

                # Skip if no faces
                if face_count == 0:
                    continue

                # Process each face in the frame
                for face in first_frame_faces:
                    # Extract bounding box coordinates
                    if 'bbox' in face:
                        x1, y1, x2, y2 = face['bbox']
                        width = x2 - x1
                        height = y2 - y1

                        # Calculate max dimension and its ratio to 1080
                        max_dimension = max(width, height)
                        ratio = max_dimension / 1080.0

                        # Add to appropriate list based on face count
                        face_ratios_by_count[face_count].append(ratio)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return face_ratios_by_count


def plot_histograms_by_face_count(face_ratios_by_count):
    """
    Plot histograms for different face counts (one-shot, two-shot, etc.) as subfigures
    Groups higher face counts into buckets (5-10 and 10+)
    """
    # Create grouped data
    grouped_data = {}

    # Individual plots for 1-4 shots
    for count in range(1, 5):
        if count in face_ratios_by_count:
            grouped_data[f"{count}-shot"] = face_ratios_by_count[count]

    # Group 5-10 shots
    five_to_ten = []
    for count in range(5, 11):
        if count in face_ratios_by_count:
            five_to_ten.extend(face_ratios_by_count[count])
    if five_to_ten:
        grouped_data["5-10 shots"] = five_to_ten

    # Group 10+ shots
    ten_plus = []
    for count in face_ratios_by_count:
        if count > 10:
            ten_plus.extend(face_ratios_by_count[count])
    if ten_plus:
        grouped_data["10+ shots"] = ten_plus

    # Colors for different groups
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'orange', 'gold']

    # Calculate number of rows and columns for subplots
    n_plots = len(grouped_data)
    n_cols = min(3, n_plots)  # Maximum 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Calculate global min and max for consistent bins across subplots
    all_ratios = [r for ratios in grouped_data.values() for r in ratios]
    min_ratio = min(all_ratios) if all_ratios else 0
    max_ratio = max(all_ratios) if all_ratios else 1
    bins = np.linspace(min_ratio, max_ratio, 30)

    # Plot each histogram in its own subplot
    for i, (label, ratios) in enumerate(grouped_data.items()):
        if i < len(axes):  # Safety check
            ax = axes[i]
            ax.hist(ratios, bins=bins, alpha=0.7,
                    color=colors[i % len(colors)],
                    edgecolor='black')

            ax.set_title(f"{label} (n={len(ratios)})")
            ax.set_xlabel('Face Size Ratio (max dimension / 1080)')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)

            # Add statistics to each subplot
            stats_text = (
                f"Mean: {np.mean(ratios):.3f}\n"
                f"Median: {np.median(ratios):.3f}"
            )
            ax.text(0.65, 0.85, stats_text, transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add overall statistics as text
    all_ratios = [r for ratios in grouped_data.values() for r in ratios]
    stats_text = (
        f"Total faces: {len(all_ratios)}\n"
        f"Mean ratio: {np.mean(all_ratios):.3f}\n"
        f"Median ratio: {np.median(all_ratios):.3f}\n"
        f"Min ratio: {np.min(all_ratios):.3f}\n"
        f"Max ratio: {np.max(all_ratios):.3f}"
    )
    plt.figtext(0.75, 0.75, stats_text, bbox=dict(
        facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('face_size_histogram_combined.png')
    # Comment out plt.show() to avoid blocking execution
    # plt.show()


if __name__ == "__main__":
    # Get face size ratios
    face_ratios_by_count = get_face_size_ratios_by_count()

    if not face_ratios_by_count:
        print("No face data found. Check the path to your face detection JSON files.")
    else:
        print(
            f"Found face data for {len(face_ratios_by_count)} different face counts.")

        # Plot histograms
        plot_histograms_by_face_count(face_ratios_by_count)
        print("Plot saved as 'face_size_histogram_combined.png'")

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from pathlib import Path
import random


def load_image(image_path):
    """Load an image from path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img


def yolov8_face_detection(model, image):
    """Detect faces using YOLOv8n-face model."""
    start_time = time.time()
    results = model(image)
    inference_time = time.time() - start_time

    # Extract bounding boxes
    boxes = []
    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, _ = box
            if conf > 0.5:  # Confidence threshold
                boxes.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

    return boxes, inference_time


def buffalo_face_detection(model, image):
    """Detect faces using InsightFace Buffalo model."""
    start_time = time.time()
    faces = model.get(image)
    inference_time = time.time() - start_time

    # Extract bounding boxes, detection scores, and pose information
    boxes = []
    face_info = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        conf = face.det_score

        if conf < 0.5:  # Skip faces with low confidence
            continue

        boxes.append([x1, y1, x2, y2, conf])

        # Extract pose information (pitch, yaw, roll)
        pose = None
        if hasattr(face, 'pose'):
            pose = face.pose

        face_info.append({
            'bbox': bbox,
            'det_score': conf,
            'pose': pose
        })

    return boxes, face_info, inference_time


def draw_boxes(image, boxes, color=(0, 255, 0)):
    """Draw bounding boxes on the image."""
    img_draw = image.copy()
    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_draw, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_draw


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def match_boxes(reference_boxes, comparison_boxes, threshold=0.5):
    """Match boxes between reference and comparison sets based on IoU."""
    matches = []
    unmatched_reference = list(range(len(reference_boxes)))
    unmatched_comparison = list(range(len(comparison_boxes)))

    # Calculate IoU for all possible pairs
    for i, ref_box in enumerate(reference_boxes):
        for j, comp_box in enumerate(comparison_boxes):
            iou = calculate_iou(ref_box, comp_box)
            if iou >= threshold:
                matches.append((i, j, iou))

    # Sort matches by IoU (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)

    # Assign matches greedily
    final_matches = []
    for ref_idx, comp_idx, iou in matches:
        if ref_idx in unmatched_reference and comp_idx in unmatched_comparison:
            final_matches.append((ref_idx, comp_idx, iou))
            unmatched_reference.remove(ref_idx)
            unmatched_comparison.remove(comp_idx)

    return final_matches, unmatched_reference, unmatched_comparison


def calculate_bbox_difference(box1, box2):
    """Calculate the difference between two bounding boxes."""
    # Calculate center points
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2

    # Calculate center point distance
    center_distance = ((center1_x - center2_x) ** 2 +
                       (center1_y - center2_y) ** 2) ** 0.5

    # Calculate size differences
    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]
    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]

    width_diff = abs(width1 - width2) / max(width1,
                                            width2) if max(width1, width2) > 0 else 0
    height_diff = abs(height1 - height2) / max(height1,
                                               height2) if max(height1, height2) > 0 else 0

    return {
        'center_distance': center_distance,
        'width_diff_pct': width_diff * 100,
        'height_diff_pct': height_diff * 100
    }


def calculate_pose_difference(pose1, pose2):
    """Calculate the difference between two pose vectors."""
    if pose1 is None or pose2 is None:
        return None

    # Calculate absolute differences for pitch, yaw, roll
    pitch_diff = abs(pose1[0] - pose2[0])
    yaw_diff = abs(pose1[1] - pose2[1])
    roll_diff = abs(pose1[2] - pose2[2])

    return {
        'pitch_diff': pitch_diff,
        'yaw_diff': yaw_diff,
        'roll_diff': roll_diff,
        'avg_diff': (pitch_diff + yaw_diff + roll_diff) / 3
    }


def filter_faces(boxes, image, min_height_ratio=0.1, blur_threshold=100.0):
    """Filter faces based on size and blurriness."""
    img_height = image.shape[0]
    min_face_height = img_height * min_height_ratio

    filtered_boxes = []
    rejected_reasons = []  # Track why faces are rejected

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        face_height = y2 - y1

        # Check if face is large enough
        if face_height < min_face_height:
            rejected_reasons.append(("too_small", box))
            continue

        filtered_boxes.append(box)

    # Uncomment for debugging
    # if len(rejected_reasons) > 0:
    #     print(f"Rejected {len(rejected_reasons)} faces: {rejected_reasons}")

    return filtered_boxes


def main():
    # Initialize models
    print("Loading YOLOv8n-face model...")
    yolo_model = YOLO(".models/yolov8n-face.pt")

    print("Loading Buffalo-L model...")
    buffalo_l_model = FaceAnalysis(name="buffalo_l", providers=[
        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    buffalo_l_model.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading Buffalo-S model...")
    buffalo_s_model = FaceAnalysis(name="buffalo_s", providers=[
        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    buffalo_s_model.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading AntelopeV2 model...")
    antelopev2_model = FaceAnalysis(name="antelopev2", providers=[
        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    antelopev2_model.prepare(ctx_id=0, det_size=(640, 640))

    # Find frames
    base_dir = "output/frames"
    frame_paths = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_masked.jpg"):
                frame_paths.append(os.path.join(root, file))

    if not frame_paths:
        print(f"No frames found in {base_dir}")
        return

    print(f"Found {len(frame_paths)} frames")

    # Process a subset of frames for comparison
    num_frames = min(100, len(frame_paths))  # Process up to 200 frames

    # Randomly select frames with a fixed seed for reproducibility
    random.seed(42)  # Set seed for reproducibility
    selected_frames = random.sample(frame_paths, num_frames)

    print(f"Randomly selected {num_frames} frames for comparison")

    # Warm-up phase
    print("Warming up models...")
    if len(selected_frames) > 0:
        warmup_image = load_image(selected_frames[0])
        # Run each model a few times to warm up
        for _ in range(3):
            _ = yolov8_face_detection(yolo_model, warmup_image)
            _ = buffalo_face_detection(buffalo_l_model, warmup_image)
            _ = buffalo_face_detection(buffalo_s_model, warmup_image)
            _ = buffalo_face_detection(antelopev2_model, warmup_image)
    print("Warm-up complete")

    # Performance metrics
    yolo_times = []
    buffalo_l_times = []
    buffalo_s_times = []
    antelopev2_times = []

    # Comparison metrics
    yolo_vs_antelope_metrics = {
        'iou_values': [],
        'center_distances': [],
        'width_diffs': [],
        'height_diffs': [],
        'matched_faces': 0,
        'total_antelope_faces': 0,
        'total_yolo_faces': 0
    }

    buffalo_l_vs_antelope_metrics = {
        'iou_values': [],
        'center_distances': [],
        'width_diffs': [],
        'height_diffs': [],
        'det_score_diffs': [],
        'pose_diffs': [],
        'matched_faces': 0,
        'total_antelope_faces': 0,
        'total_buffalo_faces': 0
    }

    buffalo_s_vs_antelope_metrics = {
        'iou_values': [],
        'center_distances': [],
        'width_diffs': [],
        'height_diffs': [],
        'det_score_diffs': [],
        'pose_diffs': [],
        'matched_faces': 0,
        'total_antelope_faces': 0,
        'total_buffalo_faces': 0
    }

    # Process frames
    for i, frame_path in enumerate(selected_frames):
        print(f"Processing frame {i+1}/{len(selected_frames)}: {frame_path}")

        # Load image
        image = load_image(frame_path)

        # YOLOv8 detection
        yolo_boxes, yolo_time = yolov8_face_detection(yolo_model, image)
        yolo_times.append(yolo_time)

        # Filter small and blurry faces
        yolo_boxes = filter_faces(yolo_boxes, image)

        # Buffalo-L detection
        buffalo_l_boxes, buffalo_l_info, buffalo_l_time = buffalo_face_detection(
            buffalo_l_model, image)
        buffalo_l_times.append(buffalo_l_time)

        # Filter small and blurry faces
        filtered_buffalo_l_boxes = filter_faces(buffalo_l_boxes, image)
        # Update face_info to match filtered boxes
        filtered_buffalo_l_info = [buffalo_l_info[i] for i, box in enumerate(buffalo_l_boxes)
                                   if box in filtered_buffalo_l_boxes]
        buffalo_l_boxes = filtered_buffalo_l_boxes
        buffalo_l_info = filtered_buffalo_l_info

        # Buffalo-S detection
        buffalo_s_boxes, buffalo_s_info, buffalo_s_time = buffalo_face_detection(
            buffalo_s_model, image)
        buffalo_s_times.append(buffalo_s_time)

        # Filter small and blurry faces
        filtered_buffalo_s_boxes = filter_faces(buffalo_s_boxes, image)
        # Update face_info to match filtered boxes
        filtered_buffalo_s_info = [buffalo_s_info[i] for i, box in enumerate(buffalo_s_boxes)
                                   if box in filtered_buffalo_s_boxes]
        buffalo_s_boxes = filtered_buffalo_s_boxes
        buffalo_s_info = filtered_buffalo_s_info

        # AntelopeV2 detection (reference model)
        antelopev2_boxes, antelopev2_info, antelopev2_time = buffalo_face_detection(
            antelopev2_model, image)
        antelopev2_times.append(antelopev2_time)

        # Filter small and blurry faces
        filtered_antelopev2_boxes = filter_faces(antelopev2_boxes, image)
        # Update face_info to match filtered boxes
        filtered_antelopev2_info = [antelopev2_info[i] for i, box in enumerate(antelopev2_boxes)
                                    if box in filtered_antelopev2_boxes]
        antelopev2_boxes = filtered_antelopev2_boxes
        antelopev2_info = filtered_antelopev2_info

        # Compare YOLOv8 with AntelopeV2
        yolo_vs_antelope_metrics['total_antelope_faces'] += len(
            antelopev2_boxes)
        yolo_vs_antelope_metrics['total_yolo_faces'] += len(yolo_boxes)

        matches, _, _ = match_boxes(antelopev2_boxes, yolo_boxes)
        yolo_vs_antelope_metrics['matched_faces'] += len(matches)

        for ref_idx, comp_idx, iou in matches:
            yolo_vs_antelope_metrics['iou_values'].append(iou)

            bbox_diff = calculate_bbox_difference(
                antelopev2_boxes[ref_idx], yolo_boxes[comp_idx])
            yolo_vs_antelope_metrics['center_distances'].append(
                bbox_diff['center_distance'])
            yolo_vs_antelope_metrics['width_diffs'].append(
                bbox_diff['width_diff_pct'])
            yolo_vs_antelope_metrics['height_diffs'].append(
                bbox_diff['height_diff_pct'])

        # Compare Buffalo-L with AntelopeV2
        buffalo_l_vs_antelope_metrics['total_antelope_faces'] += len(
            antelopev2_boxes)
        buffalo_l_vs_antelope_metrics['total_buffalo_faces'] += len(
            buffalo_l_boxes)

        matches, _, _ = match_boxes(antelopev2_boxes, buffalo_l_boxes)
        buffalo_l_vs_antelope_metrics['matched_faces'] += len(matches)

        for ref_idx, comp_idx, iou in matches:
            buffalo_l_vs_antelope_metrics['iou_values'].append(iou)

            bbox_diff = calculate_bbox_difference(
                antelopev2_boxes[ref_idx], buffalo_l_boxes[comp_idx])
            buffalo_l_vs_antelope_metrics['center_distances'].append(
                bbox_diff['center_distance'])
            buffalo_l_vs_antelope_metrics['width_diffs'].append(
                bbox_diff['width_diff_pct'])
            buffalo_l_vs_antelope_metrics['height_diffs'].append(
                bbox_diff['height_diff_pct'])

            # Compare detection scores
            det_score_diff = abs(
                antelopev2_info[ref_idx]['det_score'] - buffalo_l_info[comp_idx]['det_score'])
            buffalo_l_vs_antelope_metrics['det_score_diffs'].append(
                det_score_diff)

            # Compare pose if available
            if antelopev2_info[ref_idx]['pose'] is not None and buffalo_l_info[comp_idx]['pose'] is not None:
                pose_diff = calculate_pose_difference(
                    antelopev2_info[ref_idx]['pose'],
                    buffalo_l_info[comp_idx]['pose']
                )
                if pose_diff:
                    buffalo_l_vs_antelope_metrics['pose_diffs'].append(
                        pose_diff['avg_diff'])

        # Compare Buffalo-S with AntelopeV2
        buffalo_s_vs_antelope_metrics['total_antelope_faces'] += len(
            antelopev2_boxes)
        buffalo_s_vs_antelope_metrics['total_buffalo_faces'] += len(
            buffalo_s_boxes)

        matches, _, _ = match_boxes(antelopev2_boxes, buffalo_s_boxes)
        buffalo_s_vs_antelope_metrics['matched_faces'] += len(matches)

        for ref_idx, comp_idx, iou in matches:
            buffalo_s_vs_antelope_metrics['iou_values'].append(iou)

            bbox_diff = calculate_bbox_difference(
                antelopev2_boxes[ref_idx], buffalo_s_boxes[comp_idx])
            buffalo_s_vs_antelope_metrics['center_distances'].append(
                bbox_diff['center_distance'])
            buffalo_s_vs_antelope_metrics['width_diffs'].append(
                bbox_diff['width_diff_pct'])
            buffalo_s_vs_antelope_metrics['height_diffs'].append(
                bbox_diff['height_diff_pct'])

            # Compare detection scores
            det_score_diff = abs(
                antelopev2_info[ref_idx]['det_score'] - buffalo_s_info[comp_idx]['det_score'])
            buffalo_s_vs_antelope_metrics['det_score_diffs'].append(
                det_score_diff)

            # Compare pose if available
            if antelopev2_info[ref_idx]['pose'] is not None and buffalo_s_info[comp_idx]['pose'] is not None:
                pose_diff = calculate_pose_difference(
                    antelopev2_info[ref_idx]['pose'],
                    buffalo_s_info[comp_idx]['pose']
                )
                if pose_diff:
                    buffalo_s_vs_antelope_metrics['pose_diffs'].append(
                        pose_diff['avg_diff'])

        # Draw results
        yolo_result = draw_boxes(
            image, yolo_boxes, color=(0, 255, 0))  # Green for YOLO
        buffalo_l_result = draw_boxes(
            image, buffalo_l_boxes, color=(0, 0, 255))  # Red for Buffalo-L
        buffalo_s_result = draw_boxes(
            image, buffalo_s_boxes, color=(255, 0, 0))  # Blue for Buffalo-S
        antelopev2_result = draw_boxes(
            image, antelopev2_boxes, color=(255, 255, 0))  # Yellow for AntelopeV2

        # Save results
        output_dir = Path("output/face_detection_comparison")
        output_dir.mkdir(exist_ok=True, parents=True)

        base_name = os.path.basename(frame_path).replace("_masked.jpg", "")

        # Create 2x2 grid comparison
        h, w = image.shape[:2]
        comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        comparison[:h, :w] = yolo_result                # Top-left: YOLOv8
        comparison[:h, w:] = buffalo_l_result           # Top-right: Buffalo-L
        # Bottom-left: Buffalo-S
        comparison[h:, :w] = buffalo_s_result
        # Bottom-right: AntelopeV2
        comparison[h:, w:] = antelopev2_result

        # Add labels
        cv2.putText(comparison, f"YOLOv8n-face: {1/yolo_time:.2f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Buffalo-L: {1/buffalo_l_time:.2f} FPS", (w+10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, f"Buffalo-S: {1/buffalo_s_time:.2f} FPS", (10, h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(comparison, f"AntelopeV2: {1/antelopev2_time:.2f} FPS", (w+10, h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imwrite(
            str(output_dir / f"{base_name}_comparison.jpg"), comparison)

    # Calculate and print performance metrics
    yolo_avg_time = sum(yolo_times) / len(yolo_times)
    buffalo_l_avg_time = sum(buffalo_l_times) / len(buffalo_l_times)
    buffalo_s_avg_time = sum(buffalo_s_times) / len(buffalo_s_times)
    antelopev2_avg_time = sum(antelopev2_times) / len(antelopev2_times)

    print("\nPerformance Comparison:")
    print(
        f"YOLOv8n-face: {1/yolo_avg_time:.2f} FPS (avg inference: {yolo_avg_time*1000:.2f} ms)")
    print(
        f"Buffalo-L: {1/buffalo_l_avg_time:.2f} FPS (avg inference: {buffalo_l_avg_time*1000:.2f} ms)")
    print(
        f"Buffalo-S: {1/buffalo_s_avg_time:.2f} FPS (avg inference: {buffalo_s_avg_time*1000:.2f} ms)")
    print(
        f"AntelopeV2: {1/antelopev2_avg_time:.2f} FPS (avg inference: {antelopev2_avg_time*1000:.2f} ms)")

    # Calculate and print comparison metrics
    print("\nComparison with AntelopeV2 (Reference Model):")

    # YOLOv8 vs AntelopeV2
    print("\nYOLOv8n-face vs AntelopeV2:")
    print(f"Face detection rate: {yolo_vs_antelope_metrics['matched_faces']}/{yolo_vs_antelope_metrics['total_antelope_faces']} " +
          f"({yolo_vs_antelope_metrics['matched_faces']/yolo_vs_antelope_metrics['total_antelope_faces']*100:.2f}% of AntelopeV2 faces)")

    if yolo_vs_antelope_metrics['iou_values']:
        print(
            f"Average IoU: {sum(yolo_vs_antelope_metrics['iou_values'])/len(yolo_vs_antelope_metrics['iou_values']):.4f}")
        print(
            f"Average center distance: {sum(yolo_vs_antelope_metrics['center_distances'])/len(yolo_vs_antelope_metrics['center_distances']):.2f} pixels")
        print(
            f"Average width difference: {sum(yolo_vs_antelope_metrics['width_diffs'])/len(yolo_vs_antelope_metrics['width_diffs']):.2f}%")
        print(
            f"Average height difference: {sum(yolo_vs_antelope_metrics['height_diffs'])/len(yolo_vs_antelope_metrics['height_diffs']):.2f}%")

    # Buffalo-L vs AntelopeV2
    print("\nBuffalo-L vs AntelopeV2:")
    print(f"Face detection rate: {buffalo_l_vs_antelope_metrics['matched_faces']}/{buffalo_l_vs_antelope_metrics['total_antelope_faces']} " +
          f"({buffalo_l_vs_antelope_metrics['matched_faces']/buffalo_l_vs_antelope_metrics['total_antelope_faces']*100:.2f}% of AntelopeV2 faces)")

    if buffalo_l_vs_antelope_metrics['iou_values']:
        print(
            f"Average IoU: {sum(buffalo_l_vs_antelope_metrics['iou_values'])/len(buffalo_l_vs_antelope_metrics['iou_values']):.4f}")
        print(
            f"Average center distance: {sum(buffalo_l_vs_antelope_metrics['center_distances'])/len(buffalo_l_vs_antelope_metrics['center_distances']):.2f} pixels")
        print(
            f"Average width difference: {sum(buffalo_l_vs_antelope_metrics['width_diffs'])/len(buffalo_l_vs_antelope_metrics['width_diffs']):.2f}%")
        print(
            f"Average height difference: {sum(buffalo_l_vs_antelope_metrics['height_diffs'])/len(buffalo_l_vs_antelope_metrics['height_diffs']):.2f}%")
        print(
            f"Average detection score difference: {sum(buffalo_l_vs_antelope_metrics['det_score_diffs'])/len(buffalo_l_vs_antelope_metrics['det_score_diffs']):.4f}")

        if buffalo_l_vs_antelope_metrics['pose_diffs']:
            print(
                f"Average pose difference: {sum(buffalo_l_vs_antelope_metrics['pose_diffs'])/len(buffalo_l_vs_antelope_metrics['pose_diffs']):.2f} degrees")

    # Buffalo-S vs AntelopeV2
    print("\nBuffalo-S vs AntelopeV2:")
    print(f"Face detection rate: {buffalo_s_vs_antelope_metrics['matched_faces']}/{buffalo_s_vs_antelope_metrics['total_antelope_faces']} " +
          f"({buffalo_s_vs_antelope_metrics['matched_faces']/buffalo_s_vs_antelope_metrics['total_antelope_faces']*100:.2f}% of AntelopeV2 faces)")

    if buffalo_s_vs_antelope_metrics['iou_values']:
        print(
            f"Average IoU: {sum(buffalo_s_vs_antelope_metrics['iou_values'])/len(buffalo_s_vs_antelope_metrics['iou_values']):.4f}")
        print(
            f"Average center distance: {sum(buffalo_s_vs_antelope_metrics['center_distances'])/len(buffalo_s_vs_antelope_metrics['center_distances']):.2f} pixels")
        print(
            f"Average width difference: {sum(buffalo_s_vs_antelope_metrics['width_diffs'])/len(buffalo_s_vs_antelope_metrics['width_diffs']):.2f}%")
        print(
            f"Average height difference: {sum(buffalo_s_vs_antelope_metrics['height_diffs'])/len(buffalo_s_vs_antelope_metrics['height_diffs']):.2f}%")
        print(
            f"Average detection score difference: {sum(buffalo_s_vs_antelope_metrics['det_score_diffs'])/len(buffalo_s_vs_antelope_metrics['det_score_diffs']):.4f}")

        if buffalo_s_vs_antelope_metrics['pose_diffs']:
            print(
                f"Average pose difference: {sum(buffalo_s_vs_antelope_metrics['pose_diffs'])/len(buffalo_s_vs_antelope_metrics['pose_diffs']):.2f} degrees")

    # Plot FPS comparison
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(yolo_times)+1),
             [1/t for t in yolo_times], 'g-', label='YOLOv8n-face')
    plt.plot(range(1, len(buffalo_l_times)+1),
             [1/t for t in buffalo_l_times], 'r-', label='Buffalo-L')
    plt.plot(range(1, len(buffalo_s_times)+1),
             [1/t for t in buffalo_s_times], 'b-', label='Buffalo-S')
    plt.plot(range(1, len(antelopev2_times)+1),
             [1/t for t in antelopev2_times], 'y-', label='AntelopeV2')
    plt.xlabel('Frame Number')
    plt.ylabel('FPS')
    plt.title('Face Detection Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(output_dir / "fps_comparison.png"))

    # Plot IoU comparison
    plt.figure(figsize=(10, 6))
    if yolo_vs_antelope_metrics['iou_values']:
        plt.hist(yolo_vs_antelope_metrics['iou_values'],
                 alpha=0.5, label='YOLOv8n-face', bins=20, range=(0, 1))
    if buffalo_l_vs_antelope_metrics['iou_values']:
        plt.hist(buffalo_l_vs_antelope_metrics['iou_values'],
                 alpha=0.5, label='Buffalo-L', bins=20, range=(0, 1))
    if buffalo_s_vs_antelope_metrics['iou_values']:
        plt.hist(buffalo_s_vs_antelope_metrics['iou_values'],
                 alpha=0.5, label='Buffalo-S', bins=20, range=(0, 1))
    plt.xlabel('IoU with AntelopeV2')
    plt.ylabel('Count')
    plt.title('Bounding Box IoU Comparison with AntelopeV2')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(output_dir / "iou_comparison.png"))

    # Plot detection score comparison for InsightFace models
    plt.figure(figsize=(10, 6))
    if buffalo_l_vs_antelope_metrics['det_score_diffs']:
        plt.hist(buffalo_l_vs_antelope_metrics['det_score_diffs'],
                 alpha=0.5, label='Buffalo-L', bins=20, range=(0, 0.5))
    if buffalo_s_vs_antelope_metrics['det_score_diffs']:
        plt.hist(buffalo_s_vs_antelope_metrics['det_score_diffs'],
                 alpha=0.5, label='Buffalo-S', bins=20, range=(0, 0.5))
    plt.xlabel('Detection Score Difference with AntelopeV2')
    plt.ylabel('Count')
    plt.title('Detection Score Difference Comparison with AntelopeV2')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(output_dir / "det_score_comparison.png"))

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

import os
import requests
from pathlib import Path
import sys


def download_file(url, save_path):
    """
    Download a file from a URL and save it to the specified path.
    Shows a progress bar during download.
    """
    print(f"Downloading {url} to {save_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    downloaded = 0

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
            downloaded += len(data)

            # Update progress bar
            done = int(50 * downloaded / total_size) if total_size > 0 else 0
            sys.stdout.write(
                f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes")
            sys.stdout.flush()

    print("\nDownload complete!")


def main():
    # Create .models directory if it doesn't exist
    models_dir = Path(".models")
    models_dir.mkdir(exist_ok=True)

    # URL for the YOLOv8 face detection model
    model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"

    # Extract filename from URL
    filename = os.path.basename(model_url)
    save_path = models_dir / filename

    # Download the model
    download_file(model_url, save_path)

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

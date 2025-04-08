# Prerequisites

## Install ffmpeg
You have to have ffmpeg & ffprobe in your path. Install them accordingly.

```
On Koa:
module load vis/FFmpeg
```

## Login to Hugging Face for downloading Gemma3 model.

(Not needed on non-NVIDIA environment)

```
huggingface-cli login
```


## Download models

```
python download_models.py
```

## Update video ids (optional)
If you want to use different videos or add more videos, you can add YouTube video ids to `video_ids.txt` file.

# Commands

```
# Create a new conda environment
conda create --name rec-a-reaction python=3.9

# Activate conda environment
conda activate rec-a-reaction

# Install packages
pip install -r requirements.txt

# Download and pre-process videos
python download_and_preprocess.py --output_dir=<path_to_output_dir> --workers=1

# Extract context-reaction pairs
python extract_context_reaction_pairs.py <path_to_output_dir>
```

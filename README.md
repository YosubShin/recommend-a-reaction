# Prerequisites

## Install ffmpeg

```
ffmpeg

On Koa:
module load vis/FFmpeg
```

# Login to Hugging Face for downloading Gemma3 model

```
huggingface-cli login
```


## Download models

```
python download_models.py
```

# Commands

```
# Activate conda environment
source activate rec-a-reaction

python download_and_preprocess.py --output_dir=/home/yosubs/koa_scratch/recommend-a-reaction --workers=1
```

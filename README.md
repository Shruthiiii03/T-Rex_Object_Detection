# T-Rex: One-Shot Object Detection API
This repository provides a lightweight implementation of the T-Rex model interface, designed for one-shot
object detection via cloud inference.

## Features
- Sends reference and target images to the T-Rex server
- Receives YOLO-style predictions via API
- Normalized `.txt` output (YOLO format)
- Supports image visualization (with bounding boxes)
- Easily integrable into external pipelines

## Quick Start
### 1. Clone the Repository
```bash
git clone https://github.com/IDEA-Research/T-Rex.git
```
> Or clone into another repo as a subfolder if using as a submodule.


### 2. Install Dependencies
Make sure Python 3.8 is installed. Then:
```bash
pip install -r requirements.txt
```
> Also ensure DDS Cloud SDK (or required client API) is accessible.


## API Token
T-Rex runs inference via a cloud API. To use it:
1. Request your **DDS Cloud API key/token** from the official repo:
https://github.com/IDEA-Research/T-Rex
2. Paste your token inside `trex_model.py` or as an environment variable.
## Usage
You can run detection by calling `trex_model.py` directly or importing its functions:
```bash
python trex_model.py
```
Inside code (example):
```python
from trex_model import run_trex_inference
results = run_trex_inference(reference_images, target_images, threshold=0.8)
```

## Output Format
The model returns predictions in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```
- Coordinates are normalized to 01
- Visualizations saved in `predictions_png/`

## Limitations
- One-shot detection only: accuracy depends heavily on visual similarity
- No segmentation support (bounding boxes only)

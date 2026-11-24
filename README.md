# Pose Painter

Pose Painter turns a person in a webcam feed into an animated character using their live pose as a guide.

The project uses YOLOv8-Pose to track the person and Stable Diffusion 1.5 with OpenPose ControlNet to generate the character. LCM-LoRA keeps image generation fast enough for an interactive experience.

## What it does

- Finds the person's pose in a webcam frame.
- Converts YOLO's 17 keypoints into the 18-point OpenPose format.
- Draws the pose as an OpenPose ControlNet image.
- Generates a matching character with Stable Diffusion.
- Keeps only the newest frames and generated images so the app does not build up a backlog.

## Requirements

- Windows
- A CUDA-capable NVIDIA GPU
- A webcam
- Python 3.10 or 3.11
- Enough VRAM for Stable Diffusion

The default output size is 512 by 512. If your GPU is limited, try lowering `GENERATION_SIZE` to 384 in `main.py`.

## Setup

Clone the repository and open it in a terminal.

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the YOLO, ControlNet, Stable Diffusion, and LCM model files.

```powershell
python main.py
```


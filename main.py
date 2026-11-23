import cv2
import numpy as np
import torch
import threading
import queue
import traceback
import time
import os

from PIL import Image
from ultralytics import YOLO
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    LCMScheduler
)

SIZE = 512
CAMERA_INDEX = 0
SEED = 12345

# 384 uses less VRAM 512 gives a larger output.
GENERATION_SIZE = 512

# Keep model weights in RAM and move them to the GPU as needed.
USE_CPU_OFFLOAD = True

# Optional reference image for a consistent character.
USE_REFERENCE_IMAGE = False
REFERENCE_IMAGE_PATH = "character.png"

PROMPT = (
    "anime illustration, solo, one young adult man, "
    "full body, short hair, blue hoodie, blue jeans, "
    "consistent character design, clean anime line art, "
    "simple white background, detailed face"
)

NEGATIVE_PROMPT = (
    "female, girl, woman, multiple people, duplicate limbs, extra fingers, "
    "deformed anatomy, blurry, low quality, "
    "complex background, cropped body"
)

# Model loading requires a CUDA enabled GPU.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Install CUDA-enabled PyTorch."
    )

DEVICE = "cuda"
DTYPE = torch.float16

properties = torch.cuda.get_device_properties(0)
vram_gb = properties.total_memory / (1024 ** 3)

print("=" * 50)
print("GPU:", torch.cuda.get_device_name(0))
print(f"VRAM: {vram_gb:.2f} GB")
print("=" * 50)

print("Loading ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=DTYPE
)

print("Loading Stable Diffusion...")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None
)

# Use LCM for four step image generation.
pipe.scheduler = LCMScheduler.from_config(
    pipe.scheduler.config
)

pipe.load_lora_weights(
    "latent-consistency/lcm-lora-sdv1-5"
)

pipe.fuse_lora()

reference_image = None

if USE_REFERENCE_IMAGE:

    if not os.path.isfile(REFERENCE_IMAGE_PATH):
        raise FileNotFoundError(
            f"Reference image missing: {REFERENCE_IMAGE_PATH}"
        )

    reference_image = Image.open(
        REFERENCE_IMAGE_PATH
    ).convert("RGB")

    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin"
    )

    pipe.set_ip_adapter_scale(0.5)

# Enable memory saving options supported by this diffusers version.
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
elif hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()

if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()

# Offload model weights to CPU between GPU operations.
if USE_CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to(DEVICE)

print("Loading YOLO...")

yolo = YOLO("yolov8n-pose.pt")

print("Models loaded.")

# Keep only the latest frame in each queue.
pose_queue = queue.Queue(maxsize=1)
output_queue = queue.Queue(maxsize=1)

stop_event = threading.Event()

def put_latest(q, item):
    # Replace an older queued item with the latest one.
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            pass
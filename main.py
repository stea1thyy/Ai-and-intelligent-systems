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

# 384 uses less VRAM 512 gives a larger result.
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

# CUDA is required for model loading.
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

# LCM handles the four step generation.
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

# Turn on memory saving options supported by diffusers.
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
elif hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()

if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()

# Move model weights between CPU and GPU to keep VRAM free.
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


# Adds padding instead of stretching or squashing the frame.
def letterbox(image, size=512):

    h, w = image.shape[:2]

    scale = min(size / w, size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h)
    )

    canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    x = (size - new_w) // 2
    y = (size - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas

# OpenPose uses 18 keypoints, while YOLO gives 17.
# The neck point below is calculated from the shoulders.

# OpenPose BODY_18 indexes:
# 0 nose, 1 neck, 2 right shoulder,
# 3 right elbow, 4 right wrist,
# 5 left shoulder, 6 left elbow,
# 7 left wrist, 8 right hip,
# 9 right knee, 10 right ankle,
# 11 left hip, 12 left knee,
# 13 left ankle, 14 right eye,
# 15 left eye, 16 right ear,
# 17 left ear.

COCO_TO_OPENPOSE = {
    0: 0,
    2: 6,
    3: 8,
    4: 10,
    5: 5,
    6: 7,
    7: 9,
    8: 12,
    9: 14,
    10: 16,
    11: 11,
    12: 13,
    13: 15,
    14: 2,
    15: 1,
    16: 4,
    17: 3
}

OPENPOSE_PAIRS = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17)
]

# OpenPose pose colors used by the ControlNet model.
OPENPOSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85)
]

def convert_to_openpose(xy, confidence):

    points = np.zeros(
        (18, 2),
        dtype=np.float32
    )

    scores = np.zeros(
        18,
        dtype=np.float32
    )

    for openpose_idx, coco_idx in COCO_TO_OPENPOSE.items():

        points[openpose_idx] = xy[coco_idx]
        scores[openpose_idx] = confidence[coco_idx]

    # YOLO has no neck keypoint, so we use the shoulder midpoint.
    left_shoulder = xy[5]
    right_shoulder = xy[6]

    if confidence[5] >= 0.35 and confidence[6] >= 0.35:

        points[1] = (
            left_shoulder + right_shoulder
        ) / 2

        scores[1] = min(
            confidence[5],
            confidence[6]
        )

    return points, scores

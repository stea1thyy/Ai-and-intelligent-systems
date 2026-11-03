import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image

# Use the GPU when available, otherwise fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading models... (This will download about 2.5GB the first time, please wait)")

# 1. Load the ControlNet model (trained specifically on OpenPose skeletons)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16,
)

# 2. Load Stable Diffusion 1.5 with the ControlNet attached
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
)

# 3. Apply LCM-LoRA for extreme speed (4 steps instead of 25)
# The PEFT backend is required for loading LoRA weights in newer diffusers versions.
try:
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    print("Loaded LCM-LoRA for faster generation.")
except ValueError as exc:
    print(f"LoRA skipped: {exc}")
    print("Install the missing PEFT backend with: .\\venv\\Scripts\\python.exe -m pip install peft")

# 4. Move the pipeline to the available device and enable memory-saving options.
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
pipe.to(device)

print("Models loaded! Generating image...")

# 5. Load your skeleton image
skeleton_path = "skeleton_output.jpg"
if not os.path.exists(skeleton_path):
    raise FileNotFoundError(f"Missing skeleton input: {skeleton_path}")

skeleton_img = Image.open(skeleton_path)

# 6. Generate the image
image = pipe(
    prompt="anime character, studio ghibli style, masterpiece, highly detailed",
    image=skeleton_img,
    num_inference_steps=4,
    guidance_scale=1.0,
    controlnet_conditioning_scale=1.0,
).images[0]

# 7. Save the result
image.save("anime_character.png")
print("Success! Image saved as anime_character.png")
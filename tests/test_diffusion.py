import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image

# This ControlNet is trained to follow OpenPose skeletons.
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16
)

# Attach ControlNet to Stable Diffusion 1.5.
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
)

# LCM and its LoRA let us generate an image in just a few steps.
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()

# Reduce VAE memory use, then move the pipeline to the GPU.
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
else:
    pipe.vae.enable_slicing()
pipe.to("cuda")

print("Generating...")

# Use the pose image saved by skeleton.py.
skeleton_img = Image.open("skeleton_output.jpg")

# LCM works best with guidance_scale set to 1.
image = pipe(
    prompt="masterpiece, best quality, anime character wearing a winter jacket and long pants, full body",
    image=skeleton_img,
    num_inference_steps=4,
    guidance_scale=1.0,
    controlnet_conditioning_scale=1.0
).images[0]

# Save the generated image.
image.save("anime_character.png")
print("Done. Saved as anime_character.png")
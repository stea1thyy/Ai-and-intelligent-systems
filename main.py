import cv2
import numpy as np
import torch
import threading
import queue
from ultralytics import YOLO
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler

# Load the image generation pipeline and pose detector before opening the webcam.
print("Loading models... (Grab a coffee, this takes a sec)")
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
else:
    pipe.vae.enable_slicing()
# Run image generation on the NVIDIA GPU.
pipe.to("cuda")

yolo_model = YOLO('yolov8n-pose.pt')
print("Models loaded! Starting webcam...")

# Keep only the newest pose and generated image so slow inference cannot build a backlog.
pose_queue = queue.Queue(maxsize=1)  
output_queue = queue.Queue(maxsize=1) 

# COCO pose keypoint pairs used to draw the detected person's skeleton.
skeleton_pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), 
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), 
    (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(keypoints):
    # Convert the detected keypoints into the 512x512 ControlNet input image.
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    for pair in skeleton_pairs:
        pt1 = (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1]))
        pt2 = (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1]))
        if pt1 != (0, 0) and pt2 != (0, 0):
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 3)
    for pt in keypoints:
        x, y = int(pt[0]), int(pt[1])
        if x != 0 and y != 0:
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
    return canvas

# Generate images on a worker thread so the webcam preview stays responsive.
def diffusion_worker():
    while True:
        # Wait for a new skeleton to arrive
        skeleton_img = pose_queue.get() 
        pil_img = Image.fromarray(cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB))
        
        # Generate the character
        result = pipe(
            prompt="masterpiece, best quality, anime character wearing a cyberpunk jacket, neon lighting",
            image=pil_img,
            num_inference_steps=4,
            guidance_scale=1.0,
        ).images[0]
        
        # Drop a stale result if needed, then publish the newest generated image.
        try: output_queue.get_nowait()
        except queue.Empty: pass
        output_queue.put(np.array(result))

# Start image generation independently from webcam capture and display.
threading.Thread(target=diffusion_worker, daemon=True).start()

# Capture webcam frames, detect a pose, and display the latest results.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Resize to match the AI's expected input size
    frame_resized = cv2.resize(frame, (512, 512))
    results = yolo_model(frame_resized, verbose=False)
    
    # If YOLO finds a person
    if len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        skeleton = draw_skeleton(keypoints)
        
        # Push the newest skeleton to the AI thread
        try: pose_queue.get_nowait()
        except queue.Empty: pass
        pose_queue.put(skeleton)
        
        # Show the stick figure
        cv2.imshow("Live Pose", skeleton)
    
    # If the AI has finished a new frame, show it
    if not output_queue.empty():
        ai_image = output_queue.get()
        cv2.imshow("AI Vtuber", cv2.cvtColor(ai_image, cv2.COLOR_RGB2BGR))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
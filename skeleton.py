import cv2
import numpy as np
from ultralytics import YOLO

# YOLO expects a consistent frame size for pose detection.
image_path = 'test_person.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (512, 512))

# Get the detected body joints.
model = YOLO('yolov8n-pose.pt')
results = model(img)
keypoints = results[0].keypoints.xy[0].cpu().numpy()

# Draw the pose on a blank image.
canvas = np.zeros((512, 512, 3), dtype=np.uint8)

# Each pair marks two joints to connect.
skeleton_pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Face
    (5, 6),                               # Shoulders
    (5, 7), (7, 9),                       # Left arm
    (6, 8), (8, 10),                      # Right arm
    (5, 11), (6, 12),                     # Torso
    (11, 12),                             # Hips
    (11, 13), (13, 15),                   # Left leg
    (12, 14), (14, 16)                    # Right leg
]

for pair in skeleton_pairs:
    pt1 = (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1]))
    pt2 = (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1]))
    
    # YOLO uses (0, 0) for joints it couldn't locate.
    if pt1 != (0, 0) and pt2 != (0, 0):
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 3)

for i, pt in enumerate(keypoints):
    x, y = int(pt[0]), int(pt[1])
    if x != 0 and y != 0:
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)

# Save the pose image for the diffusion script.
cv2.imwrite('skeleton_output.jpg', canvas)
print("Skeleton saved as skeleton_output.jpg! Open it to check.")
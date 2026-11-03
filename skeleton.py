import cv2
import numpy as np
from ultralytics import YOLO

# Load the image and resize it to 512x512 
image_path = 'test_person.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (512, 512))

# Run YOLO to get the keypoints
model = YOLO('yolov8n-pose.pt')
results = model(img)
keypoints = results[0].keypoints.xy[0].cpu().numpy() # Get the X,Y coordinates

# Create a black canvas
canvas = np.zeros((512, 512, 3), dtype=np.uint8)

# Define how the joints connect
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

# Draw the lines (limbs)
for pair in skeleton_pairs:
    pt1 = (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1]))
    pt2 = (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1]))
    
    # Ignore lines if the joint goes off screen (0,0) or has low confidence
    if pt1 != (0, 0) and pt2 != (0, 0):
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 3) # Draw white line

# Draw the joints (dots)
for i, pt in enumerate(keypoints):
    x, y = int(pt[0]), int(pt[1])
    if x != 0 and y != 0:
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1) # Draw green dot

# Save the result
cv2.imwrite('skeleton_output.jpg', canvas)
print("Skeleton saved as skeleton_output.jpg! Open it to check.")
import os
import cv2
import numpy as np
import moviepy.editor as mp
import torch
import torchvision.transforms as transforms
from rembg import remove
from tqdm import tqdm  # Progress bar for frame processing

# Define directories
pwd = "/home/ubuntu/public/PUBLIC_cool_computer_vision"
sourceVideoDir = pwd + "/source"
outputDir = pwd + "/output"

# Ensure output directory exists
os.makedirs(outputDir, exist_ok=True)

# Function to apply edge detection (Sketch Effect)
def edge_detection(video_path, output_path):
    print(f"Processing Edge Detection for {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    for _ in tqdm(range(total_frames), desc="Edge Detection Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        out.write(edges)

    cap.release()
    out.release()
    print(f"Edge Detection complete: {output_path}")

# Function to remove background (Rotoscoping)
def background_removal(video_path, output_path):
    print(f"Processing Background Removal for {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Background Removal Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        no_bg = remove(frame)
        out.write(no_bg)

    cap.release()
    out.release()
    print(f"Background Removal complete: {output_path}")

# Function to slow down video using Optical Flow
def slow_motion(video_path, output_path, factor=2):
    print(f"Processing Slow Motion for {video_path}...")
    
    clip = mp.VideoFileClip(video_path)
    slow_clip = clip.fx(mp.vfx.speedx, 1/factor)

    print(f"Rendering Slow Motion Video...")
    slow_clip.write_videofile(output_path, codec="libx264", fps=clip.fps // factor)
    
    print(f"Slow Motion complete: {output_path}")

# Function to apply Neural Style Transfer (Artistic Effect)
def apply_style_transfer(video_path, output_path):
    print(f"Processing Style Transfer for {video_path}...")

    model = torch.hub.load("pytorch/examples", "fast_neural_style", model="mosaic")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Style Transfer Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze()
        styled_frame = output_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)
        out.write(styled_frame)

    cap.release()
    out.release()
    print(f"Style Transfer complete: {output_path}")

# Apply effects to all videos in the source directory
for file in os.listdir(sourceVideoDir):
    if file.endswith(".mp4"):
        video_path = os.path.join(sourceVideoDir, file)
        base_name = os.path.splitext(file)[0]

        edge_detection(video_path, f"{outputDir}/{base_name}_edges.mp4")
        background_removal(video_path, f"{outputDir}/{base_name}_no_bg.mp4")
        slow_motion(video_path, f"{outputDir}/{base_name}_slow.mp4", factor=2)
        apply_style_transfer(video_path, f"{outputDir}/{base_name}_styled.mp4")

print("All video effects applied successfully!")

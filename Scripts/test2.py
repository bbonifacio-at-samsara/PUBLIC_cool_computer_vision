import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Define directories
pwd = "/home/ubuntu/public/PUBLIC_cool_computer_vision"
sourceVideoDir = os.path.join(pwd, "source")
outputDir = os.path.join(pwd, "output")

# Ensure output directory exists
os.makedirs(outputDir, exist_ok=True)

"""
# Skip this part for now, since it's causing issues with the remote PyTorch Examples hubconf.

def apply_style_transfer(video_path, output_path, model_name="mosaic"):
    print(f"Applying Style Transfer ({model_name}) to {video_path}...")
    # (Leaving function body commented out)
"""

def cartoon_filter(video_path, output_path):
    print(f"Applying Cartoon Effect to {video_path}...")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Cartoon Effect Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        out.write(cartoon)

    cap.release()
    out.release()
    print(f"Cartoon Effect complete: {output_path}")

def glitch_effect(video_path, output_path):
    print(f"Applying Glitch Effect to {video_path}...")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Glitch Effect Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        shift = np.random.randint(-5, 5)
        glitched = np.roll(frame, shift, axis=1)
        out.write(glitched)

    cap.release()
    out.release()
    print(f"Glitch Effect complete: {output_path}")

def thermal_effect(video_path, output_path):
    print(f"Applying Thermal Vision Effect to {video_path}...")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Thermal Effect Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        out.write(thermal)

    cap.release()
    out.release()
    print(f"Thermal Vision Effect complete: {output_path}")

def motion_blur(video_path, output_path):
    print(f"Applying Motion Blur Effect to {video_path}...")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kernel_size = 10
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    for _ in tqdm(range(total_frames), desc="Motion Blur Progress", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        blurred = cv2.filter2D(frame, -1, kernel)
        out.write(blurred)

    cap.release()
    out.release()
    print(f"Motion Blur Effect complete: {output_path}")

# Process all .mp4 files in source directory
for file in os.listdir(sourceVideoDir):
    if file.endswith(".mp4"):
        video_path = os.path.join(sourceVideoDir, file)
        base_name = os.path.splitext(file)[0]

        # Skip style transfer call
        # apply_style_transfer(video_path, f"{outputDir}/{base_name}_styled.mp4", model_name="mosaic")

        cartoon_filter(video_path, f"{outputDir}/{base_name}_cartoon.mp4")
        glitch_effect(video_path, f"{outputDir}/{base_name}_glitch.mp4")
        thermal_effect(video_path, f"{outputDir}/{base_name}_thermal.mp4")
        motion_blur(video_path, f"{outputDir}/{base_name}_motion_blur.mp4")

print("All new effects applied successfully (skipping style transfer).")

import cv2
import numpy as np
from tqdm import tqdm

def calculate_checksum(frame, special_pixels):
    # Calculate checksum excluding the special pixels
    mask = np.ones(frame.shape[:2], dtype=bool)  # Create a mask for the frame
    for pixel in special_pixels:
        row_index = pixel // frame.shape[1]
        col_index = pixel % frame.shape[1]
        mask[row_index, col_index] = False  # Exclude special pixels from the mask
    
    # Calculate checksum based on the masked frame
    return np.sum(frame[mask]) % 256

def extract_checksum(frame, special_pixel):
    row_index = special_pixel // frame.shape[1]
    col_index = special_pixel % frame.shape[1]
    return frame[row_index, col_index, 0]

def decode_video(input_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fixed special pixels evenly distributed across the frame
    special_pixels = np.linspace(0, width * height - 1, 256, dtype=int)

    tampered_frames = []

    for i in tqdm(range(frame_count), desc="Decoding Video"):
        ret, frame = cap.read()
        if not ret:
            break

        embedded_checksum = extract_checksum(frame, special_pixels[i % len(special_pixels)])
        calculated_checksum = calculate_checksum(frame, special_pixels)

        # Check for mismatch
        if embedded_checksum != calculated_checksum:
            tampered_frames.append(i)

    cap.release()

    # Output the tampered frames
    if tampered_frames:
        print("Video has been tampered with. Tampered frames:")
        for frame_number in tampered_frames:
            timestamp = frame_number / fps
            print(f"Frame {frame_number} at {timestamp:.2f} seconds")
    else:
        print("The video is authentic.")

# Example usage
input_video = 'secure_input_video.mp4'
decode_video(input_video)

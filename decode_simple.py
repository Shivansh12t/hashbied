import cv2
import numpy as np
from tqdm import tqdm

def calculate_checksum(frame):
    return np.sum(frame) % 256

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

    # Calculate 256 evenly spread out pixel positions
    special_pixels = np.linspace(1000, width * height - 1, 256, dtype=int)

    tampered_frames = []

    for i in tqdm(range(frame_count), desc="Decoding Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Randomly select a special pixel to extract checksum
        special_pixel = np.random.choice(special_pixels)
        embedded_checksum = extract_checksum(frame, special_pixel)
        calculated_checksum = calculate_checksum(frame)

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

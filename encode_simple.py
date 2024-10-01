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

def embed_checksum(frame, special_pixel, checksum):
    row_index = special_pixel // frame.shape[1]
    col_index = special_pixel % frame.shape[1]
    # Embed the checksum in the specified pixel (R channel)
    frame[row_index, col_index, 0] = checksum

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Fixed special pixels evenly distributed across the frame
    special_pixels = np.linspace(0, width * height - 1, 256, dtype=int)

    for _ in tqdm(range(frame_count), desc="Encoding Video"):
        ret, frame = cap.read()
        if not ret:
            break

        original_checksum = calculate_checksum(frame, special_pixels)

        # Randomly select a special pixel for embedding
        special_pixel = special_pixels[_ % len(special_pixels)]
        embed_checksum(frame, special_pixel, original_checksum)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Encoding complete. Video saved as {output_path}")

# Example usage
input_video = 'input_video.mp4'
output_video = 'secure_' + input_video
process_video(input_video, output_video)
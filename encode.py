import cv2
import hashlib
import numpy as np
from tqdm import tqdm

def calculate_checksum(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    total_sum = np.sum(frame) - np.sum(frame[row_indices, col_indices, :])
    return total_sum % 256

def calculate_frame_hash(frame):
    frame_data = frame.tobytes()
    return np.frombuffer(hashlib.sha256(frame_data).digest(), dtype=np.uint8)

def embed_info_in_pixels(frame, special_pixels, source_id, prev_frame_hash, validity, checksum):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    
    frame[row_indices[:256], col_indices[:256], 0] = np.tile(source_id, 8)[:256]
    frame[row_indices[256:512], col_indices[256:512], 0] = np.tile(prev_frame_hash, 8)[:256]
    frame[row_indices[512:768], col_indices[512:768], 0] = np.tile(validity, 256)[:256]
    frame[row_indices[768:1024], col_indices[768:1024], 0] = np.tile(checksum, 256)[:256]

def process_video(video_path, source_id):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('secure_' + video_path, fourcc, fps, (width, height))

    prev_frame_hash = np.zeros(32, dtype=np.uint8)
    special_pixels = np.linspace(10000, width * height - 1, 1024, dtype=int)

    for frame_num in tqdm(range(frame_count), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        validity = np.array([calculate_checksum(frame, special_pixels)], dtype=np.uint8)
        checksum = validity.copy()

        embed_info_in_pixels(frame, special_pixels, source_id, prev_frame_hash, validity, checksum)

        # Log information for the last frame
        if frame_num == frame_count - 1:
            print(f"Last frame ({frame_num}):")
            print(f"Source ID: {source_id}")
            print(f"Previous frame hash: {prev_frame_hash}")
            print(f"Validity: {validity}")
            print(f"Checksum: {checksum}")

        prev_frame_hash = calculate_frame_hash(frame)
        out.write(frame)

    cap.release()
    out.release()
    print("Processing complete. Video saved as secure_" + video_path)

source_id = np.frombuffer(hashlib.sha256(b"Source Identifier").digest(), dtype=np.uint8)[:32]
process_video('input_video.mp4', source_id)
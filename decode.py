import cv2
import hashlib
import numpy as np
from tqdm import tqdm

def calculate_checksum(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    total_sum = np.sum(frame) - np.sum(frame[row_indices, col_indices, :])
    return total_sum % 256

def extract_info_from_pixels(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]

    source_id = frame[row_indices[:256], col_indices[:256], 0][:32]  # Take only first 32 values
    prev_frame_hash = frame[row_indices[256:512], col_indices[256:512], 0][:32]  # Take only first 32 values
    validity = frame[row_indices[512:768], col_indices[512:768], 0][0]  # Take only first value
    checksum = frame[row_indices[768:1024], col_indices[768:1024], 0][0]  # Take only first value

    return source_id, prev_frame_hash, validity, checksum

def calculate_frame_hash(frame):
    frame_data = frame.tobytes()
    return np.frombuffer(hashlib.sha256(frame_data).digest(), dtype=np.uint8)

def decode_video(video_path, original_source_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    special_pixels = np.linspace(10000, width * height - 1, 1024, dtype=int)
    tampered_frames = []
    prev_frame = None

    for i in tqdm(range(frame_count), desc="Validating Video"):
        ret, frame = cap.read()
        if not ret:
            break

        source_id, embedded_prev_hash, validity, checksum = extract_info_from_pixels(frame, special_pixels)
        calculated_checksum = calculate_checksum(frame, special_pixels)

        if not np.array_equal(original_source_id, source_id):
            tampered_frames.append((i, "Source ID mismatch"))
        if validity != checksum:
            tampered_frames.append((i, "Validity checksum mismatch"))
        if calculated_checksum != checksum:
            tampered_frames.append((i, "Checksum mismatch"))

        if prev_frame is not None:
            prev_frame_hash = calculate_frame_hash(prev_frame)
            if not np.array_equal(prev_frame_hash, embedded_prev_hash):
                tampered_frames.append((i, "Previous frame hash mismatch"))

        prev_frame = frame

    if tampered_frames:
        print("Video has been tampered with. Tampered frames and reasons:")
        for frame_number, reason in tampered_frames:
            timestamp = frame_number / fps
            print(f"Frame {frame_number} at {timestamp:.2f} seconds: {reason}")
    else:
        print("The video is authentic.")

original_source_id = np.frombuffer(hashlib.sha256(b"Source Identifier").digest(), dtype=np.uint8)[:32]
decode_video('secure_input_video.mp4', original_source_id)
import cv2
import numpy as np
import hashlib
from tqdm import tqdm

# Function to calculate hash of a frame
def calculate_frame_hash(frame):
    frame_data = frame.tobytes()  # Convert the frame to bytes
    return hashlib.sha256(frame_data).digest()[:32]  # Return 256-bit (32 bytes) hash

# Function to extract information embedded in 1024 specific pixels
def extract_info_from_pixels(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]

    source_id = frame[row_indices[:256], col_indices[:256]].flatten()
    prev_frame_hash = frame[row_indices[256:512], col_indices[256:512]].flatten()
    validity = frame[row_indices[512:768], col_indices[512:768]].flatten()
    checksum = frame[row_indices[768:], col_indices[768:]].flatten()

    return bytes(source_id), bytes(prev_frame_hash), int(validity[0]), int(checksum[0])

# Helper function to calculate checksum of a frame, excluding special pixels
def calculate_checksum(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    total_sum = np.sum(frame) - np.sum(frame[row_indices, col_indices, :])
    return total_sum % 256  # Return checksum (modulo 256 for simplicity)

# Function to dynamically determine the special pixels (security pixels)
def get_special_pixels(frame_height, frame_width, offset=10000):
    total_pixels = frame_height * frame_width
    special_pixel_count = 1024
    # Ensure the offset doesn't exceed the number of pixels in the frame
    offset = min(offset, total_pixels - special_pixel_count)
    return np.linspace(offset, total_pixels - 1, special_pixel_count, dtype=int)

# Main function to decode and validate video
def decode_video(video_path, original_source_id):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Retrieve FPS and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Unable to retrieve FPS from the video. Please check the video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize necessary variables
    prev_frame_hash = b'\x00' * 32  # Zero hash for the first frame
    tampered_frames = []

    # Process the video frame by frame
    for frame_number in tqdm(range(frame_count), desc="Validating Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Dynamically determine special pixels (adjust offset if it exceeds frame bounds)
        special_pixels = get_special_pixels(frame.shape[0], frame.shape[1], offset=10000)

        # Extract embedded information
        source_id, embedded_prev_hash, validity, checksum = extract_info_from_pixels(frame, special_pixels)

        # Compare the extracted source_id and previous frame hash to detect tampering
        if source_id != original_source_id:
            tampered_frames.append((frame_number, "Source ID mismatch"))
        if embedded_prev_hash != prev_frame_hash:
            tampered_frames.append((frame_number, "Previous frame hash mismatch"))

        # Calculate validity checksum (excluding special pixels)
        calculated_validity = calculate_checksum(frame, special_pixels)
        if calculated_validity != validity:
            tampered_frames.append((frame_number, "Validity checksum mismatch"))

        # Calculate checksum for the pixels excluding the first 787 and special pixels
        frame_excluding_special_and_first_787 = frame.reshape(-1)[:(1920 * 1080) - 787]
        calculated_checksum = np.sum(frame_excluding_special_and_first_787) % 256
        if calculated_checksum != checksum:
            tampered_frames.append((frame_number, "Checksum mismatch"))

        # Update the previous frame hash
        prev_frame_hash = calculate_frame_hash(frame)

    # Output the tampered frames and reasons
    if tampered_frames:
        print("Video has been tampered with. Tampered frames and reasons:")
        for frame_number, reason in tampered_frames:
            timestamp = frame_number / fps
            print(f"Frame {frame_number} at {timestamp:.2f} seconds: {reason}")
    else:
        print("Video is authentic and untampered.")

    # Release the video capture object
    cap.release()

# Example usage
original_source_id = hashlib.sha256(b"Source Identifier").digest()  # Example original source ID
decode_video('secure_input_video.mp4', original_source_id)

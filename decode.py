import cv2
import hashlib
import numpy as np
from tqdm import tqdm

# Helper function to calculate checksum of a frame, excluding the special pixels
def calculate_checksum(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    total_sum = np.sum(frame) - np.sum(frame[row_indices, col_indices, :])
    return total_sum % 256  # Return checksum (modulo 256 for simplicity)

# Function to extract embedded information from special pixels
def extract_info_from_pixels(frame, special_pixels):
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]

    # Extract the embedded data
    source_id = frame[row_indices[0:256], col_indices[0:256]]
    prev_frame_hash = frame[row_indices[256:512], col_indices[256:512]]
    validity = frame[row_indices[512:768], col_indices[512:768]]
    checksum = frame[row_indices[768:1024], col_indices[768:1024]]

    return source_id.flatten(), prev_frame_hash.flatten(), validity.flatten(), checksum.flatten()

# Function to calculate the hash of a frame
def calculate_frame_hash(frame):
    # Flatten the frame to a 1D array and calculate its hash
    flattened_frame = frame.flatten()
    frame_hash = hashlib.sha256(flattened_frame).digest()
    return np.frombuffer(frame_hash, dtype=np.uint8)

# Main function to decode video and validate checksums
def decode_video(video_path, original_source_id):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Cannot determine FPS of the video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Predefine the positions of the 4 security pixel blocks within the frame
    def get_special_pixels(frame_height, frame_width, offset=10000):
        total_pixels = frame_height * frame_width
        offset = min(offset, total_pixels - 1024)
        pixel_indices = np.linspace(offset, total_pixels - 1, 1024, dtype=int)
        return pixel_indices

    tampered_frames = []

    prev_frame = None

    # Process each frame
    for i in tqdm(range(frame_count), desc="Validating Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Get special pixel positions
        special_pixels = get_special_pixels(height, width)

        # Extract embedded information from special pixels
        source_id, embedded_prev_hash, validity, checksum = extract_info_from_pixels(frame, special_pixels)

        # Validate checksums and hashes
        validity_checksum = np.sum(validity) % 256
        calculated_checksum = calculate_checksum(frame, special_pixels)

        # Check for mismatches
        if not np.array_equal(original_source_id, source_id):
            tampered_frames.append((i, "Source ID mismatch"))
        if validity_checksum != checksum[0]:
            tampered_frames.append((i, "Validity checksum mismatch"))
        if calculated_checksum != checksum[0]:
            tampered_frames.append((i, "Checksum mismatch"))

        # Calculate the previous frame hash for comparison if this is not the first frame
        if prev_frame is not None:
            prev_frame_hash = calculate_frame_hash(prev_frame)
            if not np.array_equal(prev_frame_hash, embedded_prev_hash):
                tampered_frames.append((i, "Previous frame hash mismatch"))

        # Update prev_frame for the next iteration
        prev_frame = frame

    # Output the results
    if tampered_frames:
        print("Video has been tampered with. Tampered frames and reasons:")
        for frame_number, reason in tampered_frames:
            timestamp = frame_number / fps
            print(f"Frame {frame_number} at {timestamp:.2f} seconds: {reason}")
    else:
        print("The video is authentic.")

# Example usage
original_source_id = hashlib.sha256(b"Source Identifier").digest()  # Example source ID
decode_video('secure_input_video.mp4', original_source_id)

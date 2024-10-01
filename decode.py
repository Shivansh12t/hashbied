import cv2
import hashlib
import numpy as np
from tqdm import tqdm

# Helper function to calculate checksum of a frame, excluding the special pixels
def calculate_checksum(frame, special_pixels):
    # Convert linear pixel indices to 2D (row, col) indices for accessing pixel values
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    
    # Get the sum of all pixel values excluding the special pixels
    total_sum = np.sum(frame) - np.sum(frame[row_indices, col_indices, :])
    return total_sum % 256  # Return checksum (modulo 256 for simplicity)

# Helper function to calculate hash of the frame
def calculate_frame_hash(frame):
    frame_data = frame.tobytes()  # Convert the frame to bytes
    return hashlib.sha256(frame_data).digest()[:32]  # Return 256-bit (32 bytes) hash

# Function to extract embedded information from 1024 specific pixels
# Function to extract embedded information from 1024 specific pixels
def extract_info_from_pixels(frame, special_pixels):
    # Convert linear pixel indices to 2D (row, col) indices for accessing pixel values
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    
    # Extract the source ID, previous frame hash, validity, and checksum from the special pixels
    # Extracting only one channel (for example, blue channel, which is index 0)
    source_id = [frame[row_indices[i], col_indices[i], 0] for i in range(256)]
    prev_frame_hash = [frame[row_indices[i + 256], col_indices[i + 256], 0] for i in range(256)]
    validity = [frame[row_indices[i + 512], col_indices[i + 512], 0] for i in range(32)]
    checksum = [frame[row_indices[i + 768], col_indices[i + 768], 0] for i in range(32)]
    
    return bytes(source_id), bytes(prev_frame_hash), bytes(validity), bytes(checksum)


# Main function to decode video and check for tampering
def decode_video(video_path, original_source_id):
    # Open video
    cap = cv2.VideoCapture(video_path)

    # Check if the video is 1080p
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != 1920 or height != 1080:
        print("Error: Video is not 1080p.")
        return

    # Get video details
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables
    prev_frame_hash = b'\x00' * 32  # Zero hash for the first frame
    tampered_frames = []

    def get_special_pixels(frame_height, frame_width, offset=10000):
        total_pixels = frame_height * frame_width
        offset = min(offset, total_pixels - 1024)
        pixel_indices = np.linspace(offset, total_pixels - 1, 1024, dtype=int)
        return pixel_indices

    # Process video frame by frame
    for i in tqdm(range(frame_count), desc="Validating Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Determine special pixels
        special_pixels = get_special_pixels(height, width, offset=10000)

        # Extract the embedded information from the current frame
        source_id, embedded_prev_hash, validity, checksum = extract_info_from_pixels(frame, special_pixels)

        # Verify the source ID
        if source_id != original_source_id:
            tampered_frames.append((i, "Source ID mismatch"))

        # Calculate validity checksum and compare with the embedded validity
        calculated_validity = calculate_checksum(frame, special_pixels)
        if calculated_validity != int.from_bytes(validity, byteorder='big') % 256:
            tampered_frames.append((i, "Validity checksum mismatch"))

        # Verify the previous frame's hash
        if prev_frame_hash != embedded_prev_hash:
            tampered_frames.append((i, "Previous frame hash mismatch"))

        # Calculate the frame checksum and compare with the embedded checksum
        frame_excluding_special_and_first_787 = frame.reshape(-1)[:(1920 * 1080) - 787]
        calculated_checksum = np.sum(frame_excluding_special_and_first_787) % 256
        if calculated_checksum != int.from_bytes(checksum, byteorder='big') % 256:
            tampered_frames.append((i, "Frame checksum mismatch"))

        # Calculate the current frame hash for the next iteration
        prev_frame_hash = calculate_frame_hash(frame)

    # Release the video object
    cap.release()

    # Output the verdict
    if tampered_frames:
        print("Video has been tampered with. Tampered frames and reasons:")
        for frame, reason in tampered_frames:
            timestamp = frame / cap.get(cv2.CAP_PROP_FPS)
            print(f"Frame {frame}, Timestamp {timestamp:.2f}s: {reason}")
    else:
        print("The video is authentic and untampered.")

# Example usage
original_source_id = hashlib.sha256(b"Source Identifier").digest()  # Original source ID used in encoding
decode_video('secure_input_video.mp4', original_source_id)

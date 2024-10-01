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

# Function to embed information in 1024 specific pixels
def embed_info_in_pixels(frame, special_pixels, source_id, prev_frame_hash, validity, checksum):
    # Convert linear pixel indices to 2D (row, col) indices for accessing pixel values
    row_indices = special_pixels // frame.shape[1]
    col_indices = special_pixels % frame.shape[1]
    
    # Convert validity and checksum to bytes
    validity_bytes = np.array(validity, dtype=np.uint8).tobytes().ljust(32, b'\0')
    checksum_bytes = np.array(checksum, dtype=np.uint8).tobytes().ljust(32, b'\0')
    
    # Embed the source ID, previous frame hash, validity, and checksum in the special pixels
    for i in range(256):
        frame[row_indices[i], col_indices[i]] = source_id[i % len(source_id)]
        frame[row_indices[i + 256], col_indices[i + 256]] = prev_frame_hash[i % len(prev_frame_hash)]
        frame[row_indices[i + 512], col_indices[i + 512]] = validity_bytes[i % 32]
        frame[row_indices[i + 768], col_indices[i + 768]] = checksum_bytes[i % 32]

# Main function to process video and embed security pixels
def process_video(video_path, source_id):
    # Open video
    cap = cv2.VideoCapture(video_path)

    # Check if the video is 1080p
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != 1920 or height != 1080:
        print("Error: Video is not 1080p.")
        return

    # Get video details
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object to save the new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('secure_' + video_path, fourcc, fps, (1920, 1080))

    # Initialize variables
    prev_frame_hash = b'\x00' * 32  # Start with zero hash for the first frame
    special_pixel_count = 1024  # 1024 special pixels

    # Predefine the positions of the 4 security pixel blocks within the frame
    def get_special_pixels(frame_height, frame_width, offset=10000):
        total_pixels = frame_height * frame_width
        # Make sure offset doesn't exceed the number of pixels in the frame
        offset = min(offset, total_pixels - special_pixel_count)
        pixel_indices = np.linspace(offset, total_pixels - 1, special_pixel_count, dtype=int)
        return pixel_indices

    # Process video frame by frame
    for i in tqdm(range(frame_count), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Dynamically determine special pixels (adjust offset if it exceeds frame bounds)
        special_pixels = get_special_pixels(height, width, offset=10000)

        # Calculate validity checksum (excluding special pixels)
        validity = calculate_checksum(frame, special_pixels)

        # Calculate checksum of the pixels excluding the special 1024 security pixels
        # We compute the checksum for the entire frame except the first 787 pixels and the special pixels
        frame_excluding_special_and_first_787 = frame.reshape(-1)[:(1920 * 1080) - 787]
        checksum = np.sum(frame_excluding_special_and_first_787) % 256

        # Embed the source_id, previous frame hash, validity bits, and checksum
        embed_info_in_pixels(frame, special_pixels, source_id, prev_frame_hash, validity, checksum)

        # Calculate hash of the current frame
        prev_frame_hash = calculate_frame_hash(frame)

        # Write the processed frame to the new video file
        out.write(frame)

    # Release the video objects
    cap.release()
    out.release()
    print("Processing complete. Video saved as secure_" + video_path)

# Example usage
source_id = hashlib.sha256(b"Source Identifier").digest()  # Example source ID (256 bits)
process_video('input_video.mp4', source_id)

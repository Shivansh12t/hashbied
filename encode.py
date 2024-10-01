import cv2
import hashlib
import numpy as np
from tqdm import tqdm

# Helper function to calculate checksum of a frame
def calculate_checksum(frame, special_pixels):
    # Get the sum of all pixel values excluding the special pixels
    total_sum = np.sum(frame) - np.sum(frame[special_pixels])
    return total_sum % 256  # Return checksum (modulo 256 for simplicity)

# Helper function to calculate hash of the frame
def calculate_frame_hash(frame):
    frame_data = frame.tobytes()  # Convert the frame to bytes
    return hashlib.sha256(frame_data).digest()[:32]  # Return 256-bit (32 bytes) hash

# Function to embed information in 64 specific pixels
def embed_info_in_pixels(frame, special_pixels, source_id, prev_frame_hash, validity, checksum):
    frame[special_pixels[0]] = source_id[:32]  # First 256 bits for Source ID (assuming 32 bytes)
    frame[special_pixels[1]] = prev_frame_hash[:32]  # Next 256 bits for Previous Frame Hash
    frame[special_pixels[2]] = validity.to_bytes(32, 'big')  # Validity bits (32 bytes)
    frame[special_pixels[3]] = checksum.to_bytes(32, 'big')  # Checksum (32 bytes)

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
    special_pixels_offset = 0  # 10,000 pixel offset
    special_pixel_count = 1024  # 1024 special pixels

    # Predefine the positions of the 4 security pixel blocks within the 1024 special pixels
    special_pixels = np.linspace(special_pixels_offset, 1920*1080, special_pixel_count, dtype=int).reshape(4, 256)

    # Process video frame by frame
    for i in tqdm(range(frame_count), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate validity checksum (excluding special pixels)
        validity = calculate_checksum(frame, special_pixels)

        # Calculate checksum of the first 787 pixels
        checksum = calculate_checksum(frame[:787], special_pixels)

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

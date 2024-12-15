import cv2
import numpy as np
import os
import airsim

# Define the directory where the images are saved
tmp_dir = "/home/mkg7/nanosam/assets"  # Make sure this is where your images are saved

# List of image filenames to load
image_files = [
    os.path.join(tmp_dir, "0.pfm"),  # DepthVis image
    os.path.join(tmp_dir, "1.pfm"),  # DepthPerspective image
    os.path.join(tmp_dir, "2.png"),  # Scene image in PNG format
    os.path.join(tmp_dir, "3.png")   # Uncompressed Scene image
]

# Function to load and return the NumPy array from PFM files (depth images)
def load_pfm(filename):
    try:
        depth_data, scale = airsim.read_pfm(filename)  # PFM returns both data and scale
        return np.array(depth_data, dtype=np.float32)  # Convert to NumPy array with float32 type
    except Exception as e:
        print(f"Failed to load PFM file {filename}: {e}")
        return None

# Function to display images
def display_images(image_files):
    depth_image = load_pfm(image_files[1])  # Load DepthPerspective
    scene_image = cv2.imread(image_files[2])  # Load Scene image

    if depth_image is None or scene_image is None:
        print("Failed to load necessary images.")
        return

    # Normalize the depth image for visualization
    normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth = np.uint8(normalized_depth)

    # Apply a color map to the normalized depth image
    depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

    # Combine the depth color map with the scene image
    # You can adjust the alpha for transparency (0.5 here means 50% transparency)
    alpha = 0.9
    combined_image = cv2.addWeighted(scene_image, 1 - alpha, depth_colormap, alpha, 0)

    # Display the combined segmented image
    cv2.imshow("Segmented Image", combined_image)

    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()

# Call the function to display the segmented images
display_images(image_files)


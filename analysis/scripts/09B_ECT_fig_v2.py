#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from skimage import measure # Needed for finding contours

###########################
### CONFIGURATION ###
###########################

# Define the project root by navigating up from the script's location.
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

# Path to the final prepared dataset from the data generation script.
FINAL_PREPARED_DATA_FILE = PROJECT_ROOT_DIR / "outputs" / "synthetic_leaf_data" / "final_cnn_dataset.pkl"

# Output directory and filename for the visualization figure
FIGURES_OUTPUT_DIR = PROJECT_ROOT_DIR / "outputs" / "figures"
FIGURE_OUTPUT_PATH = FIGURES_OUTPUT_DIR / "fig_ECT.png"

# Ensure the output directory exists
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

# Adjusted layout for 123 classes
NUM_ROWS = 12
NUM_COLS = 11
FIGURE_WIDTH_INCHES = 8.5
FIGURE_HEIGHT_INCHES = 11

###########################
### DATA LOADING ###
###########################

print(f"--- Loading data from {FINAL_PREPARED_DATA_FILE} ---")
try:
    with open(FINAL_PREPARED_DATA_FILE, 'rb') as f:
        final_data = pickle.load(f)

    X_images = final_data['X_images'] # (N, H, W, 2) numpy array
    y_labels_encoded = final_data['y_labels_encoded'] # (N,) numpy array
    is_real_flags = final_data['is_real_flags'] # (N,) boolean numpy array
    class_names = final_data['class_names'] # List of string class names
    
    if len(class_names) > NUM_ROWS * NUM_COLS:
        print(f"Warning: The number of classes ({len(class_names)}) exceeds the panel count ({NUM_ROWS * NUM_COLS}). Some classes may not be displayed.")
    else:
        print(f"Loaded {len(class_names)} classes, which will fit into a {NUM_ROWS}x{NUM_COLS} grid.")

    # Recreate LabelEncoder from class_names for inverse_transform functionality
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    # Find the indices of real samples
    real_sample_indices = np.where(is_real_flags)[0]

    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"Error: Data file not found at {FINAL_PREPARED_DATA_FILE}.")
    print("Please ensure the data generation script has been run successfully.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

################################
### FIGURE GENERATION ###
################################

print(f"\n--- Generating figure with {len(class_names)} panels ---")

# Create a figure with the specified size and grid layout
fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), squeeze=False)
axes = axes.flatten()

# Group real samples by their encoded label
real_samples_by_class = {}
for idx in real_sample_indices:
    label = y_labels_encoded[idx]
    if label not in real_samples_by_class:
        real_samples_by_class[label] = []
    real_samples_by_class[label].append(idx)

# Loop through each class and plot a representative image
for class_idx in range(len(class_names)):
    ax = axes[class_idx]
    official_name = class_names[class_idx]
    
    # Try to find a real sample for this class
    if class_idx in real_samples_by_class and len(real_samples_by_class[class_idx]) > 0:
        
        # NEW LOGIC: Iterate through leaves to find one with a good contour
        chosen_sample_idx = None
        for sample_idx in real_samples_by_class[class_idx]:
            mask_image = X_images[sample_idx, :, :, 0]
            contours = measure.find_contours(mask_image > 0, 0.5)
            # A good contour is likely a single, large one. Check if it's the only one.
            if len(contours) == 1:
                chosen_sample_idx = sample_idx
                break
        
        # If a good sample was found, use it; otherwise, default to the first sample.
        if chosen_sample_idx is None:
            chosen_sample_idx = real_samples_by_class[class_idx][0]

        # Get the ECT and mask images for the chosen sample
        ect_image = X_images[chosen_sample_idx, :, :, 1]
        mask_image = X_images[chosen_sample_idx, :, :, 0]
        
        # Plot the ECT image with the reversed grayscale colormap.
        ax.imshow(ect_image, cmap='gray_r')
        
        # Find contours of the shape mask for the chosen sample
        contours = measure.find_contours(mask_image > 0, 0.5)
        
        # Plot ALL contours that are larger than a small threshold.
        # This fixes cases where the leaf outline is broken into multiple parts.
        contour_size_threshold = 20 # Minimum number of points for a contour to be plotted
        if contours:
            for contour in contours:
                if len(contour) > contour_size_threshold:
                    ax.plot(contour[:, 1], contour[:, 0], color='black', linewidth=1.0)
        
        # Set the title to the official name
        ax.set_title(official_name, fontsize=6)
        
    else:
        # Handle case where a class has no real samples
        ax.text(0.5, 0.5, f"No Real Samples\nfor {official_name}", 
                 ha='center', va='center', fontsize=6, color='gray', wrap=True)

    # Remove axes ticks and labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

# Hide any extra subplots if the number of classes is less than the total panels
for i in range(len(class_names), len(axes)):
    fig.delaxes(axes[i])

# Adjust the layout to prevent titles from overlapping
plt.tight_layout(pad=0.5)

# Save the figure to the specified path
plt.savefig(FIGURE_OUTPUT_PATH, dpi=300)

print(f"Figure saved to {FIGURE_OUTPUT_PATH}")

# Display the plot
plt.show()

print("\n--- Script Completed ---")
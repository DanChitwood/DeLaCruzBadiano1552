import pandas as pd
import numpy as np
import math
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import os
import re

# --- Configuration and Path Setup ---
# The script expects to be in the 'scripts' directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input directories
LEAF_TRACES_DIR = PROJECT_ROOT / "data" / "leaf_traces"
METADATA_FILE = PROJECT_ROOT / "outputs" / "master_sheet_processing" / "verified_subchapter_links.csv"
LEAF_TRACE_FOLDERS = ["ale", "dan", "kylie", "lachlann", "mariana", "noe", "zoe"]
JIMENA_FOLDER = "jimena"

# Output directories and filenames
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_MORPHOMETRICS_DIR = PROJECT_ROOT / "outputs" / "morphometrics"

# Ensure output directories exist
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MORPHOMETRICS_DIR.mkdir(parents=True, exist_ok=True)

# Output filenames for saving data and plots
PCA_PARAMS_H5_FILENAME = "pca_model_parameters.h5"
PROCESSED_DATA_H5_FILENAME = "aligned_coords_scores_metadata.h5"
PCA_EXPLAINED_VARIANCE_REPORT_FILENAME = "pca_explained_variance.txt"
ALIGNMENT_CHECK_PLOT_FILENAME = "gpa_alignment_check.png"

# --- Parameters for Analysis ---
RANDOM_SEED = 42 # Set for reproducibility
np.random.seed(RANDOM_SEED)

INTERPOLATION_POINTS = 200 # Number of equidistant points for each leaf trace
NUM_DIMENSIONS = 2 # For 2D coordinates

# Plotting parameters for alignment check figure
ALIGNMENT_PLOT_WIDTH = 6
ALIGNMENT_PLOT_HEIGHT = 8
GPA_MEAN_COLOR = "black"
GPA_MEAN_LINEWIDTH = 2
GPA_MEAN_ZORDER = 10
INDIVIDUAL_OUTLINE_COLOR = "gray"
INDIVIDUAL_OUTLINE_ALPHA = 0.2
INDIVIDUAL_OUTLINE_LINEWIDTH = 0.5

print("--- Starting 07_morphometric_analysis.py ---")
print(f"Project root is: {PROJECT_ROOT}")

# --- Helper Functions ---
def interpolation(x, y, number):
    """
    Returns equally spaced, interpolated points for a given polyline.
    """
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    if distance[-1] == 0:
        return np.full(number, x[0]), np.full(number, y[0])
    distance = distance/distance[-1]
    fx, fy = interp1d(distance, x), interp1d(distance, y)
    alpha = np.linspace(0, 1, number)
    return fx(alpha), fy(alpha)

def gpa_mean(leaf_arr, landmark_num, dim_num):
    """
    Calculates the Generalized Procrustes Analysis mean shape.
    """
    ref_ind = 0
    ref_shape = leaf_arr[ref_ind, :, :]
    mean_diff = 10**(-30)
    old_mean = ref_shape
    d = 1000000
    while d > mean_diff:
        arr = np.zeros(((len(leaf_arr)), landmark_num, dim_num))
        for i in range(len(leaf_arr)):
            s1, s2, distance = procrustes(old_mean, leaf_arr[i])
            arr[i] = s2
        new_mean = np.mean(arr, axis=0)
        s1, s2, d = procrustes(old_mean, new_mean)
        old_mean = new_mean
    return new_mean

def rotate_points(xvals, yvals, degrees):
    """
    Rotates 2D x and y coordinate points around the origin.
    """
    rads = np.deg2rad(degrees)
    new_xvals = xvals * np.cos(rads) - yvals * np.sin(rads)
    new_yvals = xvals * np.sin(rads) + yvals * np.cos(rads)
    return new_xvals, new_yvals

def plot_gpa_alignment_check(mean_shape, aligned_shapes, save_path):
    """
    Generates a plot to visually check the GPA alignment.
    Shows the mean leaf in a bold outline with all other aligned leaves in a thin, transparent outline.
    """
    plt.figure(figsize=(ALIGNMENT_PLOT_WIDTH, ALIGNMENT_PLOT_HEIGHT))
    ax = plt.gca()

    # Plot all individual aligned leaves
    for i in range(aligned_shapes.shape[0]):
        ax.plot(aligned_shapes[i, :, 0], aligned_shapes[i, :, 1], 
                 color=INDIVIDUAL_OUTLINE_COLOR, 
                 alpha=INDIVIDUAL_OUTLINE_ALPHA, 
                 linewidth=INDIVIDUAL_OUTLINE_LINEWIDTH)

    # Plot the GPA mean leaf on top in a bold, black outline
    ax.plot(mean_shape[:, 0], mean_shape[:, 1], 
             color=GPA_MEAN_COLOR, 
             linewidth=GPA_MEAN_LINEWIDTH, 
             zorder=GPA_MEAN_ZORDER)
    
    ax.set_title("GPA Alignment Check: All Leaves Aligned to Mean Shape")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"GPA alignment check plot saved to {save_path}")

# --- Step 1: Data Parsing and Collection ---
print("\n--- 1/5: Loading leaf trace data ---")
plant_traces = {}
total_traces = 0

for folder_name in LEAF_TRACE_FOLDERS:
    folder_path = LEAF_TRACES_DIR / folder_name
    if not folder_path.exists(): continue
    for file_path in folder_path.glob("*.txt"):
        plant_id = file_path.stem.rsplit('_', 1)[0]
        
        # --- FIX FOR MISLABELED ID p055_1 ---
        if plant_id == "p055_1":
            plant_id = "p055_01"

        try:
            coords = np.loadtxt(file_path, delimiter='\t')
            if plant_id not in plant_traces:
                plant_traces[plant_id] = []
            plant_traces[plant_id].append(coords)
            total_traces += 1
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

jimena_path = LEAF_TRACES_DIR / JIMENA_FOLDER
if jimena_path.exists():
    for subfolder in jimena_path.iterdir():
        if subfolder.is_dir():
            plant_id = subfolder.name
            
            # --- FIX FOR MISLABELED ID p055_1 (in case it's in jimena) ---
            if plant_id == "p055_1":
                plant_id = "p055_01"

            for file_path in subfolder.glob("*.txt"):
                try:
                    coords = np.loadtxt(file_path, delimiter='\t')
                    if plant_id not in plant_traces:
                        plant_traces[plant_id] = []
                    plant_traces[plant_id].append(coords)
                    total_traces += 1
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

if not plant_traces:
    print("No leaf traces found. Exiting.")
    exit()

print(f"Found traces for {len(plant_traces)} unique plants.")
print(f"Found a total of {total_traces} individual leaf traces.")

# --- Step 2: Metadata Processing and Merging ---
print("\n--- 2/5: Processing metadata and linking to traces ---")

# Load metadata
mdata_raw = pd.read_csv(METADATA_FILE)

# Create a dictionary for quick lookup of metadata by illustration ID
id_to_metadata = {}
all_metadata_ids = set()
for _, row in mdata_raw.iterrows():
    illustrations = str(row['illustrations'])
    if illustrations.lower() != 'nan':
        # Split and strip multiple IDs in the 'illustrations' column
        plant_ids = [pid.strip() for pid in illustrations.split(';')]
        for pid in plant_ids:
            id_to_metadata[pid] = {
                'ID': row['ID'],
                'official_name': row['official_name'],
                'type': row['type'],
                'language': row['language'],
                'morphemes': row['morphemes']
            }
            all_metadata_ids.add(pid)

# Create a master list of all leaf traces found
all_trace_ids = set(plant_traces.keys())

# --- Generate the Mismatch Report ---
# The master list should contain all unique IDs from both sources
all_unique_ids = all_trace_ids.union(all_metadata_ids)

mismatch_report_data = []
for pid in sorted(list(all_unique_ids)):
    is_in_trace = pid in all_trace_ids
    is_in_metadata = pid in all_metadata_ids
    
    if is_in_trace and is_in_metadata:
        # Perfect match, get metadata and append
        metadata_row = id_to_metadata[pid]
        mismatch_report_data.append({
            'plant_id': pid,
            'match_found': True,
            'official_name': metadata_row['official_name']
        })
    elif is_in_trace:
        # Found in traces, but not in metadata
        mismatch_report_data.append({
            'plant_id': pid,
            'match_found': False,
            'official_name': None
        })
    elif is_in_metadata:
        # Found in metadata, but not in traces
        mismatch_report_data.append({
            'plant_id': pid,
            'match_found': False,
            'official_name': id_to_metadata[pid]['official_name']
        })

mismatch_report_df = pd.DataFrame(mismatch_report_data)
report_filepath = OUTPUT_MORPHOMETRICS_DIR / "mismatch_report.csv"
mismatch_report_df.to_csv(report_filepath, index=False)
print(f"Mismatch report saved to {report_filepath}")

# --- Proceed with only the matched leaves ---
processed_leaf_data = []
processed_mdata = []

for plant_id, traces in plant_traces.items():
    if plant_id in id_to_metadata:
        for i, trace in enumerate(traces):
            processed_leaf_data.append(trace)
            mdata_row = id_to_metadata[plant_id].copy()
            mdata_row['plant_id'] = plant_id
            mdata_row['trace_index'] = i
            processed_mdata.append(mdata_row)

if not processed_leaf_data:
    print("No leaves with matching metadata found. Exiting.")
    exit()

mdata = pd.DataFrame(processed_mdata)
print(f"Successfully linked metadata for {len(processed_leaf_data)} individual leaf traces.")

# --- Step 3: Interpolation, GPA, and PCA ---
print("\n--- 3/5: Interpolating, aligning, and performing PCA ---")

num_traces = len(processed_leaf_data)
lf_arr = np.zeros((num_traces, INTERPOLATION_POINTS, NUM_DIMENSIONS))

for i, coords in enumerate(processed_leaf_data):
    x_vals, y_vals = interpolation(coords[:, 0], coords[:, 1], INTERPOLATION_POINTS)
    lf_arr[i, :, :] = np.column_stack((x_vals, y_vals))

# Calculate GPA mean
mean_shape = gpa_mean(lf_arr, INTERPOLATION_POINTS, NUM_DIMENSIONS)

# --- REVISED GPA MEAN ORIENTATION (FINAL) ---
# Center the shape first
mean_shape -= np.mean(mean_shape, axis=0)

# Now find the principal axis to align the leaf vertically
pca_mean = PCA(n_components=2)
pca_mean.fit(mean_shape)
principal_axis = pca_mean.components_[0]

# Calculate the angle of the principal axis
current_angle = np.rad2deg(np.arctan2(principal_axis[1], principal_axis[0]))

# Target angle is 90 degrees (vertical)
rotation_angle = 90 - current_angle

# Rotate the mean shape to be perfectly vertical
rot_x, rot_y = rotate_points(mean_shape[:, 0], mean_shape[:, 1], rotation_angle)
rot_mean = np.column_stack((rot_x, rot_y))

# --- FINAL 180-DEGREE ROTATION FIX ---
# Based on the previous output, the leaf is pointing downwards.
# A final 180-degree rotation will flip it to point upwards.
final_rot_x, final_rot_y = rotate_points(rot_mean[:, 0], rot_mean[:, 1], 180)
rot_mean = np.column_stack((final_rot_x, final_rot_y))

# Align all leaves to this new, upright standardized mean
proc_arr = np.zeros(np.shape(lf_arr))
for i in range(num_traces):
    _, s2, _ = procrustes(rot_mean, lf_arr[i, :, :])
    proc_arr[i] = s2

# Reshape for PCA
flat_arr = proc_arr.reshape(num_traces, INTERPOLATION_POINTS * NUM_DIMENSIONS)

# Perform full PCA for variance analysis and saving
max_pc_components = min(flat_arr.shape[0], flat_arr.shape[1])
pca = PCA(n_components=max_pc_components, random_state=RANDOM_SEED)
pca.fit(flat_arr)
PCs = pca.transform(flat_arr)

print("Alignment and PCA completed.")

# --- Step 4: Saving Outputs ---
print("\n--- 4/5: Saving analysis outputs ---")

# Save PCA Explained Variance Report
report_filepath = OUTPUT_MORPHOMETRICS_DIR / PCA_EXPLAINED_VARIANCE_REPORT_FILENAME
with open(report_filepath, 'w') as f:
    f.write("PCA Explained Variance Report:\n")
    f.write(f"Total Samples: {flat_arr.shape[0]}\n")
    f.write(f"Number of PCs Calculated: {pca.n_components_}\n\n")
    for i in range(len(pca.explained_variance_ratio_)):
        pc_variance = round(pca.explained_variance_ratio_[i] * 100, 2)
        cumulative_variance = round(pca.explained_variance_ratio_.cumsum()[i] * 100, 2)
        line = f"PC{i+1}: {pc_variance}%, {cumulative_variance}%\n"
        f.write(line)
print(f"PCA explained variance report saved to {report_filepath}")

# Save PCA Model Parameters
pca_params_filepath = OUTPUT_MORPHOMETRICS_DIR / PCA_PARAMS_H5_FILENAME
with h5py.File(pca_params_filepath, 'w') as f:
    f.create_dataset('components', data=pca.components_, compression="gzip")
    f.create_dataset('mean', data=pca.mean_, compression="gzip")
    f.create_dataset('explained_variance', data=pca.explained_variance_, compression="gzip")
    f.create_dataset('explained_variance_ratio', data=pca.explained_variance_ratio_, compression="gzip")
    f.attrs['n_components'] = pca.n_components_
print(f"PCA model parameters saved to {pca_params_filepath}")

# Save aligned coords, PCA scores, and metadata
processed_data_filepath = OUTPUT_MORPHOMETRICS_DIR / PROCESSED_DATA_H5_FILENAME
with h5py.File(processed_data_filepath, 'w') as f:
    f.create_dataset('aligned_coords', data=proc_arr, compression="gzip")
    f.create_dataset('pca_scores', data=PCs, compression="gzip")
    for col in mdata.columns:
        # H5py requires string data to be encoded as bytes
        f.create_dataset(f'metadata/{col}', data=np.array(mdata[col]).astype('S'), compression="gzip")
print(f"Aligned coordinates, PCA scores, and metadata saved to {processed_data_filepath}")

# --- Step 5: Create and save GPA Alignment Check Plot ---
print("\n--- 5/5: Creating and saving GPA alignment check plot ---")

alignment_plot_filepath = OUTPUT_FIGURES_DIR / ALIGNMENT_CHECK_PLOT_FILENAME
plot_gpa_alignment_check(rot_mean, proc_arr, alignment_plot_filepath)

print("\n--- All processing and saving completed. ---")
import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# --- Configuration: All paths and file names defined here ---

# Project and Directory Structure
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Input Data Paths
LEAVES_METRICS_PATH = OUTPUTS_DIR / "cnn_analysis" / "trained_models" / "metrics_output" / "Illustrated_Plants_Morpheme_CNN_classification_report_Nahuatl_Morphemes.json"
ENGLISH_METRICS_PATH = OUTPUTS_DIR / "fasttext_cnn_analysis" / "metrics" / "english" / "FastText_Morpheme_TextCNN_classification_report_Nahuatl_Morphemes.json"
SPANISH_METRICS_PATH = OUTPUTS_DIR / "fasttext_cnn_analysis" / "metrics" / "spanish" / "FastText_Morpheme_TextCNN_classification_report_Nahuatl_Morphemes.json"

LEAVES_IMAGES_DIR = OUTPUTS_DIR / "cnn_analysis" / "trained_models" / "grad_cam_images"
ENGLISH_WC_DIR = OUTPUTS_DIR / "fasttext_cnn_analysis" / "wordcloud_figures" / "english"
SPANISH_WC_DIR = OUTPUTS_DIR / "fasttext_cnn_analysis" / "wordcloud_figures" / "spanish"

# Output File Names
TABLE_CSV_PATH = FIGURES_DIR / "table_CNN_morpheme.csv"
TABLE_TXT_PATH = FIGURES_DIR / "table_CNN_morpheme.txt"
FIGURE_PNG_PATH = FIGURES_DIR / "fig_CNN_morpheme.png"

# Morpheme Classes and Labels
MORPHEME_CLASSES = ["Xihuitl", "Xochitl", "Quahuitl", "Patli", "Quilitl"]
ROW_LABELS = MORPHEME_CLASSES
COL_LABELS = ["Leaves", "English", "Spanish"]
TABLE_ROWS = MORPHEME_CLASSES + ["micro avg", "macro avg", "weighted avg"]

# --- Helper Function to load and parse metrics ---
def load_and_parse_metrics(file_path, column_prefix):
    """Loads a JSON metrics file and extracts precision, recall, and f1-score."""
    if not file_path.exists():
        print(f"Error: Metrics file not found at {file_path}. Skipping.")
        return None
    
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    
    data = {}
    for cls in TABLE_ROWS:
        class_metrics = metrics.get(cls, {})
        data[f"Precision ({column_prefix})"] = class_metrics.get("precision", 0.0)
        data[f"Recall ({column_prefix})"] = class_metrics.get("recall", 0.0)
        data[f"F1 ({column_prefix})"] = class_metrics.get("f1-score", 0.0)
    return data

# --- 1. Generate the combined results table ---
print("--- Generating results table... ---")

# Load data from all three sources
leaves_data = load_and_parse_metrics(LEAVES_METRICS_PATH, "Leaves")
english_data = load_and_parse_metrics(ENGLISH_METRICS_PATH, "English")
spanish_data = load_and_parse_metrics(SPANISH_METRICS_PATH, "Spanish")

if not all([leaves_data, english_data, spanish_data]):
    print("Could not generate table due to missing input files.")
else:
    # Create DataFrames from the loaded data
    df_leaves = pd.DataFrame(leaves_data, index=TABLE_ROWS)
    df_english = pd.DataFrame(english_data, index=TABLE_ROWS)
    df_spanish = pd.DataFrame(spanish_data, index=TABLE_ROWS)
    
    # Concatenate the DataFrames
    combined_df = pd.concat([df_leaves, df_english, df_spanish], axis=1)

    # Transpose the DataFrame to have metrics as rows and classes as columns
    transposed_df = combined_df.T
    transposed_df.index.name = 'Metric'
    
    # Round all values to 2 decimal places
    transposed_df = transposed_df.round(2)
    
    # Save to CSV
    transposed_df.to_csv(TABLE_CSV_PATH)
    print(f"Table saved to CSV: {TABLE_CSV_PATH}")
    
    # Save to Markdown TXT
    markdown_table = transposed_df.to_markdown()
    with open(TABLE_TXT_PATH, 'w') as f:
        f.write(markdown_table)
    print(f"Table saved to Markdown TXT: {TABLE_TXT_PATH}")
    print("\n" + markdown_table) # Also print to console for quick review

# --- 2. Generate the combined figure ---
print("\n--- Generating comparative figure... ---")

# Set fixed figure size to 8.5x11 inches
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8.5, 11), dpi=300)

# Set up column titles
for ax, col in zip(axes[0], COL_LABELS):
    ax.set_title(col, fontsize=12, fontweight='bold', ha='center')

# Set up row titles closer to the subplots
for ax, row in zip(axes[:,0], ROW_LABELS):
    ax.text(-0.02, 0.5, row, transform=ax.transAxes, ha='right', va='center', fontsize=12, fontweight='bold')

# Populate subplots with images
for i, morpheme_class in enumerate(MORPHEME_CLASSES):
    # Leaves (GradCAM) are in the first column
    leaves_img_path = LEAVES_IMAGES_DIR / f"Illustrated_Plants_Morpheme_CNN_GradCAM_{morpheme_class}.png"
    if leaves_img_path.exists():
        img = mpimg.imread(leaves_img_path)
        axes[i, 0].imshow(img)
    else:
        axes[i, 0].text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=10)

    # English Word Cloud are in the second column
    english_wc_path = ENGLISH_WC_DIR / f"wordcloud_gradcam_{morpheme_class}.png"
    if english_wc_path.exists():
        img = mpimg.imread(english_wc_path)
        axes[i, 1].imshow(img)
    else:
        axes[i, 1].text(0.5, 0.5, "Word Cloud not found", ha='center', va='center', fontsize=10)

    # Spanish Word Cloud are in the third column
    spanish_wc_path = SPANISH_WC_DIR / f"wordcloud_gradcam_{morpheme_class}.png"
    if spanish_wc_path.exists():
        img = mpimg.imread(spanish_wc_path)
        axes[i, 2].imshow(img)
    else:
        axes[i, 2].text(0.5, 0.5, "Word Cloud not found", ha='center', va='center', fontsize=10)

# Turn off axes for all subplots
for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

# Adjust subplot parameters to minimize whitespace
plt.subplots_adjust(wspace=0.01, hspace=0.00, left=0.15, right=0.95, top=0.95, bottom=0.05)

plt.savefig(FIGURE_PNG_PATH)
plt.close()

print(f"Combined figure saved to: {FIGURE_PNG_PATH}")

print("\nScript completed successfully.")
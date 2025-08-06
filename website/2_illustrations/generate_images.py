import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --------------------------
# CONFIGURATION
# --------------------------

# New paths based on your instructions
data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data'))
master_csv_path = os.path.join(data_dir, "FOR NAHUATL REVIEW - Nahuatl names.csv")
synonym_csv_path = "../1_parse_text/verified_synonym_matches.csv" # Path is relative to the script's directory
pics_dir = os.path.join(data_dir, "leaf_traces", "pics")
leaves_dirs = ["ale", "dan", "jimena", "kylie", "lachlann", "mariana", "noe", "zoe"]
attribution = ["A", "D", "J", "K", "L", "M", "N", "Z"]

assets_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "3_assets"))
os.makedirs(assets_dir, exist_ok=True)

resolution_scale = 0.5
original_size = (1350, 1838)

# --------------------------
# UTILITY FUNCTIONS
# --------------------------

def extract_page_num(ill_name):
    match = re.search(r'p0*([0-9]{1,3})', ill_name)
    return int(match.group(1)) if match else None

def get_page_from_code(code):
    return code.split("_")[0]

def get_leaf_paths(illustration_code):
    matches = []
    # New loop to use the updated leaves_dirs
    for leaf_dir in leaves_dirs:
        # --- FIX STARTS HERE ---
        if leaf_dir == "jimena":
            base_path = os.path.join(data_dir, "leaf_traces", leaf_dir, illustration_code)
            if os.path.isdir(base_path):
                files = glob(os.path.join(base_path, "*.txt"))
                if files:
                    matches.append((leaf_dir, files))
        else:
            files = glob(os.path.join(data_dir, "leaf_traces", leaf_dir, f"{illustration_code}_*.txt"))
            if files:
                matches.append((leaf_dir, files))
        # --- FIX ENDS HERE ---
    return matches

def read_leaf_coords(file_path):
    try:
        return np.loadtxt(file_path)
    except:
        return None

def save_plot(image, leaf_data, save_name):
    fig, ax = plt.subplots(figsize=(original_size[0]/100, original_size[1]/100), dpi=100)
    ax.imshow(image)
    for coords in leaf_data:
        if coords is not None and coords.shape[1] == 2:
            ax.fill(coords[:, 0], coords[:, 1])
    ax.axis("off")
    full_path = os.path.join(assets_dir, save_name)
    fig.set_size_inches(original_size[0]/100 * resolution_scale, original_size[1]/100 * resolution_scale)
    fig.savefig(full_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# --------------------------
# LOAD AND CREATE DATAFRAME
# --------------------------

# Load master and synonym data
master_df = pd.read_csv(master_csv_path)
synonym_df = pd.read_csv(synonym_csv_path)

# Combine the data to replicate 'illustration_names.csv' in memory
df = master_df.copy()

# Add 'verified_subchapter' by grouping synonyms
subchapters_by_id = synonym_df.groupby('ID').apply(
    lambda x: "; ".join([f"{row['spelling']} {row['subchapter']}" if row['variant_checked'] is not None else row['subchapter'] for _, row in x.iterrows()])
)
df['verified_subchapter'] = df['ID'].map(subchapters_by_id)
df['verified_subchapter'].fillna('', inplace=True)

# Rename 'illustrations' column and add 'pages' column
df.rename(columns={'illustrations': 'illustration_name'}, inplace=True)
df['illustration_name'].fillna('', inplace=True)

# Replicate the logic from the old script to get to the `df_expanded` state
# Expand by illustration_name
df['illustration_name'] = df['illustration_name'].astype(str).str.split(';').apply(lambda lst: [i.strip() for i in lst])
df = df.explode('illustration_name').reset_index(drop=True)

# Drop rows with no illustration name
df = df[df['illustration_name'] != ''].reset_index(drop=True)

# Extract page numbers from illustration_name
df['pages'] = df['illustration_name'].apply(extract_page_num)

# Expand verified illustration into multiple rows and clean (re-using old logic)
expanded_rows = []
for _, row in df.iterrows():
    # As 'verified illustration' doesn't exist, we use 'illustration_name' to build the rows.
    # The 'illustration_name' column from master_df contains the info
    entry = str(row.get("illustration_name", "")).strip()

    if entry == "" or entry.isnumeric():
        name = None
        page = entry if entry.isnumeric() else ""
    else:
        parts = entry.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            name, page = parts[0], parts[1]
        else:
            name, page = None, entry

    new_row = row.copy()
    new_row["verified illustration"] = entry
    new_row["spelling"] = (name or row["official_name"]).lower()
    new_row["page"] = page
    expanded_rows.append(new_row)

df_expanded = pd.DataFrame(expanded_rows)

# Drop intermediate column if still present
if "pages" in df_expanded.columns:
    df_expanded.drop(columns=["pages"], inplace=True)

# --------------------------
# IMAGE CREATION BY NAME
# --------------------------

output_rows = []

for idx, row in df_expanded.iterrows():
    illustration_code = row["illustration_name"]
    page_code = get_page_from_code(illustration_code)
    pic_path = os.path.join(pics_dir, f"{page_code}.png")

    if os.path.exists(pic_path):
        leaf_matches = get_leaf_paths(illustration_code)
        if leaf_matches:
            try:
                img = plt.imread(pic_path)
            except:
                img = None

            if img is not None:
                for leaf_dir, paths in leaf_matches:
                    coords_list = [read_leaf_coords(p) for p in paths]
                    idx_leaf_dir = leaves_dirs.index(leaf_dir)
                    attr_code = attribution[idx_leaf_dir]
                    official_name_clean = row["official_name"].replace(" ", "_")
                    save_name = f"{attr_code}_{row['ID']}_{illustration_code}_{official_name_clean}.png"
                    save_plot(img, coords_list, save_name)

                    new_row = row.copy()
                    new_row["filename"] = save_name
                    output_rows.append(new_row)

df_output = pd.DataFrame(output_rows)
df_output = df_output.drop_duplicates(subset=["ID", "illustration_name"])

# --------------------------
# IMAGE CREATION BY SUBCHAPTER
# --------------------------

unique_pages = df_output["illustration_name"].dropna().apply(get_page_from_code).unique()

for page_code in unique_pages:
    pic_path = os.path.join(pics_dir, f"{page_code}.png")
    if not os.path.exists(pic_path):
        continue

    try:
        img = plt.imread(pic_path)
    except:
        continue

    leaf_matches = []
    active_attributions = set()

    for leaf_dir in leaves_dirs:
        # --- FIX STARTS HERE ---
        if leaf_dir == "jimena":
            base_path = os.path.join(data_dir, "leaf_traces", leaf_dir, page_code)
            if os.path.isdir(base_path):
                paths = glob(os.path.join(base_path, "*.txt"))
                if paths:
                    leaf_matches.extend([read_leaf_coords(p) for p in paths])
                    idx_leaf_dir = leaves_dirs.index(leaf_dir)
                    active_attributions.add(attribution[idx_leaf_dir])
        else:
            paths = glob(os.path.join(data_dir, "leaf_traces", leaf_dir, f"{page_code}_*.txt"))
            if paths:
                leaf_matches.extend([read_leaf_coords(p) for p in paths])
                idx_leaf_dir = leaves_dirs.index(leaf_dir)
                active_attributions.add(attribution[idx_leaf_dir])
        # --- FIX ENDS HERE ---

    if not leaf_matches:
        continue

    attr_code = "".join(sorted(active_attributions))
    save_name = f"{attr_code}_{page_code}.png"
    save_plot(img, leaf_matches, save_name)

# --------------------------
# EXPORT CLEANED CSV
# --------------------------

df_output.to_csv("assets.csv", index=False)
print("Finished generating images and saving CSV to assets.csv")
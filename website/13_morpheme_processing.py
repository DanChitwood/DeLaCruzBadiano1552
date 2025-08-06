import os
import shutil
import pandas as pd
from pathlib import Path

# --- Configuration ---
input_csv = "../data/FOR NAHUATL REVIEW - Nahuatl names.csv"
input_folders = [
    "6_chapters", "7_plants_with_illlustrations", "8_plants",
    "9_stones", "10_animals", "11_birds", "12_other"
]
output_folder = "14_all_md_files"
os.makedirs(output_folder, exist_ok=True)

# --- Load morpheme data from the new CSV ---
df = pd.read_csv(input_csv)
df["official_name"] = df["official_name"].astype(str)

# --- Process the 'label' column to create the morpheme dictionary ---
# Filter out rows where the 'label' column is empty, NA, or contains only "NA"
df = df[df["label"].notna() & df["label"].str.strip().str.lower().ne("na")]

# Create a dictionary of official_name to a list of morphemes
def split_labels(label_str):
    # Split by semicolon and remove leading/trailing whitespace
    return [morpheme.strip() for morpheme in str(label_str).split(';') if morpheme.strip()]

grouped_labels = df.groupby("official_name")["label"].apply(
    lambda x: sorted(list(set([item for sublist in x.apply(split_labels) for item in sublist])))
).to_dict()

# --- File processing ---
for folder in input_folders:
    input_path = Path(folder)
    output_path = Path(output_folder)

    for md_file in input_path.glob("*.md"):
        # The script attempts to get a name like "Acxoyatl" or "01"
        official_name_raw = md_file.stem.split("_")[0]

        # Use the raw name for chapters and the cleaned name for other files
        if folder == "6_chapters":
            shutil.copy(md_file, output_path / md_file.name)
            continue
        
        # Match the official name to the dictionary keys
        morphemes = grouped_labels.get(official_name_raw)

        # Read the content of the markdown file
        with open(md_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if morphemes:
            morpheme_block = ["\n", "**Morphemes:**\n", "\n"] + [f"- {m}\n" for m in morphemes] + ["\n"]

            new_lines = []
            inserted = False
            i = 0
            
            while i < len(lines):
                line = lines[i]
                new_lines.append(line)

                # Insert the morpheme block after the Variants section if it exists
                if "**Variants:**" in line and not inserted:
                    i += 1
                    new_lines.append(lines[i])
                    # Collect all bullet points after "Variants"
                    while i + 1 < len(lines) and lines[i + 1].lstrip().startswith("-"):
                        i += 1
                        new_lines.append(lines[i])
                    new_lines.append("\n") # Add a blank line
                    new_lines.extend(morpheme_block)
                    inserted = True
                
                # Insert the morpheme block before the "Subchapter" header if Variants are not found
                elif not inserted and line.strip().startswith("## Subchapter"):
                    new_lines = new_lines[:-1] + morpheme_block + [line]
                    inserted = True
                
                i += 1
            
            # If neither insertion point was found, prepend the morpheme block to the file
            if not inserted:
                new_lines = morpheme_block + lines

            with open(output_path / md_file.name, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        else:
            # If no morphemes are found, copy the file as-is
            shutil.copy(md_file, output_path / md_file.name)

print(f"Finished adding morpheme information to markdown files. Outputs are in the '{output_folder}' directory.")
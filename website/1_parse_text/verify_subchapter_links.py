import pandas as pd
import os
import re

# Load the metadata
# Updated path to the new data source
df = pd.read_csv("../../data/FOR NAHUATL REVIEW - Nahuatl names.csv")

# Path to subchapter text files
# Updated path to the new text folder
subchapter_folder = "../../data/texts"

# Helper: Normalize subchapter name (e.g., "8j" → "08j")
def normalize_subchapter(sc):
    match = re.match(r'(\d{1,2})([a-z])', sc)
    return f"{int(match.group(1)):02d}{match.group(2)}" if match else sc

from collections import defaultdict

records = []
seen_subchapter_variants = set()

for _, row in df.iterrows():
    id_ = row['ID']
    official = str(row['official_name']).strip().lower()
    
    # Use 'official_name' in place of 'nahuatl' as instructed
    nahuatl = str(row['official_name']).strip().lower() 
    typ = row['type']
    
    # Collect all possible spelling variants
    variants = set(filter(None, [official, nahuatl]))
    subchapters = []

    # Safely parse the subchapter list from the new 'text_subchapters' column
    subchapter_field = str(row.get('text_subchapters', '')).strip()
    synonym_field = str(row.get('synonyms', '')).strip()

    if not subchapter_field:
        continue  # Skip rows with no subchapter listed

    # Parse the subchapters from the 'text_subchapters' column
    parts = [p.strip() for p in subchapter_field.split(';') if p.strip()]
    for part in parts:
        subch = normalize_subchapter(part)
        subchapters.append((subch, None))
    
    # Add synonyms to the variants set
    if synonym_field:
        synonyms = [s.strip().lower() for s in synonym_field.split(';') if s.strip()]
        variants.update(synonyms)

    # For each subchapter, check for any matching variant
    for subch, _ in subchapters:
        # Look for any file that starts with the subchapter code followed by _p and ends in .txt
        matches = [f for f in os.listdir(subchapter_folder) if f.startswith(f"{subch}_p") and f.endswith(".txt")]
        if not matches:
            continue  # No matching file found
        
        # Assume there's only one match per subchapter
        txt_path = os.path.join(subchapter_folder, matches[0])

        # Skip missing files silently
        if not os.path.exists(txt_path):
            continue
        
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        found_variants = set()
        for v in variants:
            if v in content:
                found_variants.add(v)

        if found_variants:
            for fv in sorted(found_variants):
                key = (id_, subch, fv)
                if key not in seen_subchapter_variants:
                    seen_subchapter_variants.add(key)
                    records.append({
                        "ID": id_,
                        "official_name": row['official_name'],
                        "type": typ,
                        "subchapter": subch,
                        "spelling": fv,
                        "variant_checked": fv
                    })
        else:
            key = (id_, subch, "NOT FOUND")
            if key not in seen_subchapter_variants:
                seen_subchapter_variants.add(key)
                records.append({
                    "ID": id_,
                    "official_name": row['official_name'],
                    "type": typ,
                    "subchapter": subch,
                    "spelling": "NOT FOUND",
                    "variant_checked": None
                })

# Convert to DataFrame
result_df = pd.DataFrame.from_records(records)
result_df.to_csv("verified_synonym_matches.csv", index=False)
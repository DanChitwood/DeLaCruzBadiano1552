import pandas as pd
import os
import re
from collections import defaultdict

# --- Configuration ---
# Define the base project root, assuming scripts/ is current working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # This gets to '/your_project_root/scripts'
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT) # This goes up to '/your_project_root'

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# --- MODIFIED: Added subfolder for outputs ---
OUTPUTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUTS_DIR = os.path.join(OUTPUTS_BASE_DIR, "master_sheet_processing")
# --- END MODIFIED ---

MASTER_SHEET_FILENAME = "FOR NAHUATL REVIEW - Nahuatl names.csv"
MASTER_SHEET_PATH = os.path.join(DATA_DIR, MASTER_SHEET_FILENAME)

ENGLISH_TEXTS_FOLDER = os.path.join(DATA_DIR, "texts")
SPANISH_TEXTS_FOLDER = os.path.join(DATA_DIR, "texts_ES")

# Output CSV filenames
VERIFIED_LINKS_CSV = os.path.join(OUTPUTS_DIR, "verified_subchapter_links.csv")
ABSENT_LINKS_CSV = os.path.join(OUTPUTS_DIR, "absent_subchapter_links.csv")
MORPHEME_COUNTS_CSV = os.path.join(OUTPUTS_DIR, "morpheme_counts.csv")

# Ensure output directory exists (including the new subfolder)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- Helper Functions ---

def normalize_subchapter_code(sc_code: str) -> str:
    """
    Normalizes a subchapter code (e.g., "8j" -> "08j", "12a" -> "12a").
    Handles cases like "chapter01" by returning as is, assuming it's not a
    subchapter code to be normalized in this way.
    """
    match = re.match(r'(\d{1,2})([a-z])', sc_code)
    if match:
        return f"{int(match.group(1)):02d}{match.group(2)}"
    return sc_code # Return as is if no match (e.g., already 2 digits or chapter files)

def load_master_sheet(file_path: str) -> pd.DataFrame:
    """
    Loads the master sheet CSV and performs initial data cleaning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Master sheet file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Drop specified columns
    cols_to_drop = ['PROBLEMS', 'NOTES', 'REFERENCES']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Strip whitespace and handle potential NaN for string columns used in logic
    string_cols = ['official_name', 'type', 'text_subchapters', 'illustrations', 'synonyms', 'label']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('').str.strip()
            
    return df

def parse_semicolon_list(text: str) -> list[str]:
    """
    Parses a semicolon-separated string into a list of cleaned, non-empty strings.
    """
    if not text:
        return []
    return [item.strip() for item in text.split(';') if item.strip()]

def get_subchapter_text_content(subchapter_code: str, lang_type: str) -> tuple[str | None, str | None]:
    """
    Finds and reads the content of a subchapter text file for a given language.
    Assumes file names are like 'XXy_p*.txt' or 'chapterXX_p*.txt'.
    
    Args:
        subchapter_code (str): The subchapter code (e.g., '01a', 'chapter01').
        lang_type (str): 'english' or 'spanish'.
    
    Returns (content, filepath_if_found) or (None, None).
    """
    if lang_type == 'english':
        folder_path = ENGLISH_TEXTS_FOLDER
    elif lang_type == 'spanish':
        folder_path = SPANISH_TEXTS_FOLDER
    else:
        print(f"Error: Unknown language type '{lang_type}'.")
        return None, None
        
    # Determine the file search pattern based on whether it's a subchapter or chapter preface
    if subchapter_code.startswith("chapter"): # For files like chapter01_p005.txt
        search_prefix = f"{subchapter_code}_p"
    else: # For files like 01a_p007.txt
        search_prefix = f"{normalize_subchapter_code(subchapter_code)}_p"
        
    matches = [f for f in os.listdir(folder_path) 
               if f.startswith(search_prefix) and f.endswith(".txt")]
    
    if not matches:
        return None, None # No matching file found
    
    # Assume there's only one match per search prefix
    txt_filename = matches[0]
    txt_path = os.path.join(folder_path, txt_filename)

    if not os.path.exists(txt_path):
        return None, None # Should not happen if matches list is correct
        
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().lower(), txt_path # Return content in lowercase
    except Exception as e:
        print(f"Error reading file {txt_path}: {e}")
        return None, None

# --- Main Processing Logic ---
def process_master_sheet_and_texts():
    print(f"Loading master sheet from: {MASTER_SHEET_PATH}")
    master_df = load_master_sheet(MASTER_SHEET_PATH)
    print(f"Loaded {len(master_df)} rows from master sheet.")

    verified_records = []
    absent_records = []
    morpheme_counts = defaultdict(int)

    # Convert all synonyms to lowercase once for efficient searching
    master_df['synonyms_lower'] = master_df['synonyms'].apply(
        lambda x: [s.lower() for s in parse_semicolon_list(x)]
    )
    # Also include official_name (lowercase) as a search term
    master_df['official_name_lower'] = master_df['official_name'].str.lower()
    master_df['search_terms'] = master_df.apply(
        lambda row: set(row['synonyms_lower'] + [row['official_name_lower']]), axis=1
    )
    
    for index, row in master_df.iterrows():
        plant_id = row['ID']
        official_name = row['official_name']
        plant_type = row['type']
        raw_text_subchapters = row['text_subchapters']
        raw_illustrations = row['illustrations']
        raw_morphemes = row['label'] # This is the full label string

        # Update morpheme counts (do this for all rows, regardless of text matches)
        parsed_morphemes = parse_semicolon_list(raw_morphemes)
        for unit in parsed_morphemes:
            morpheme_counts[unit] += 1

        # Skip if no subchapters are linked in the master sheet for this row
        if not raw_text_subchapters:
            continue 

        linked_subchapter_codes = [
            normalize_subchapter_code(sc) for sc in parse_semicolon_list(raw_text_subchapters)
        ]
        
        # Ensure 'official_name_lower' is always part of search terms
        search_terms_for_id = list(row['search_terms'])
        # Add official name as a search term if it's not already in the synonyms list
        if row['official_name_lower'] not in search_terms_for_id:
            search_terms_for_id.append(row['official_name_lower'])


        for sc_code in linked_subchapter_codes:
            for lang_name in ['english', 'spanish']: # Iterate through languages
                subchapter_content, subchapter_filepath = get_subchapter_text_content(sc_code, lang_name)
                
                if subchapter_content is None:
                    # Record absence due to file not found
                    absent_records.append({
                        "ID": plant_id,
                        "official_name": official_name,
                        "subchapter_code_expected": sc_code,
                        "language_checked": lang_name,
                        "reason_for_absence": "subchapter_file_not_found",
                        "synonyms_expected": row['synonyms'] # Original synonyms string for context
                    })
                    continue # Move to next language/subchapter for this sc_code
                
                found_synonym_in_content = None
                for term in search_terms_for_id:
                    if term and term in subchapter_content: # Ensure term is not empty string
                        found_synonym_in_content = term
                        break # Found one, no need to check others for this (ID, subchapter, lang)
                
                if found_synonym_in_content:
                    # Record verified match
                    verified_records.append({
                        "ID": plant_id,
                        "official_name": official_name,
                        "type": plant_type,
                        "subchapter_code": sc_code,
                        "language": lang_name,
                        "synonym_found": found_synonym_in_content,
                        "morphemes": raw_morphemes,
                        "illustrations": raw_illustrations,
                        "master_sheet_subchapters_linked": raw_text_subchapters # Keeping for now, can be removed later
                    })
                else:
                    # Record absence due to synonym not found in content
                    absent_records.append({
                        "ID": plant_id,
                        "official_name": official_name,
                        "subchapter_code_expected": sc_code,
                        "language_checked": lang_name,
                        "reason_for_absence": "no_synonym_found_in_subchapter",
                        "synonyms_expected": row['synonyms'] # Original synonyms string for context
                    })

    # --- Create DataFrames and Save ---

    # Verified Links
    verified_df = pd.DataFrame(verified_records)
    print(f"Generated {len(verified_df)} verified records.")
    verified_df.to_csv(VERIFIED_LINKS_CSV, index=False)
    print(f"Saved verified links to: {VERIFIED_LINKS_CSV}")

    # Absent Links
    absent_df = pd.DataFrame(absent_records)
    print(f"Generated {len(absent_df)} absent records.")
    absent_df.to_csv(ABSENT_LINKS_CSV, index=False)
    print(f"Saved absent links to: {ABSENT_LINKS_CSV}")

    # Morpheme Counts
    morpheme_counts_df = pd.DataFrame(morpheme_counts.items(), columns=['morpheme_unit', 'count'])
    # --- MODIFIED: Sort by count in descending order ---
    morpheme_counts_df = morpheme_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    # --- END MODIFIED ---
    print(f"Generated {len(morpheme_counts_df)} unique morpheme records.")
    morpheme_counts_df.to_csv(MORPHEME_COUNTS_CSV, index=False)
    print(f"Saved morpheme counts to: {MORPHEME_COUNTS_CSV}")

    print("Processing complete!")

# --- Main Execution ---
if __name__ == "__main__":
    process_master_sheet_and_texts()
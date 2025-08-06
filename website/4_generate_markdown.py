import pandas as pd
import os
import numpy as np
import re
def add_space_after_punctuation(text):
    """Adds a single space after each period and comma in the given text."""
    text = re.sub(r'(?<=[.,])(?=[^\s])', ' ', text)
    return text

def replace_terms(text, search_terms, replacement_terms):
    synonym_map = {
        syn.lower(): rep for syn, rep in zip(search_terms, replacement_terms)
    }
    sorted_terms = sorted(synonym_map.items(), key=lambda x: len(x[0]), reverse=True)
    pattern = re.compile('|'.join(re.escape(term) for term, _ in sorted_terms), flags=re.IGNORECASE)

    result = []
    last_index = 0
    replaced_spans = []

    for match in pattern.finditer(text):
        start, end = match.span()
        if any(rs < end and start < re for rs, re in replaced_spans):
            continue

        matched_text = match.group(0)
        matched_text_lower = matched_text.lower()

        if matched_text_lower in synonym_map:
            filename = synonym_map[matched_text_lower]
            replacement = f"[{matched_text}]({filename}.md)"
        else:
            replacement = matched_text

        result.append(text[last_index:start])
        result.append(replacement)
        replaced_spans.append((start, end))
        last_index = end

    result.append(text[last_index:])
    return ''.join(result)

def bold_first_sentence(text):
    first_period_index = text.find('.')
    if first_period_index != -1:
        return f"**{text[:first_period_index + 1]}**{text[first_period_index + 1:]}"
    else:
        # If there's no period, bold the entire block
        return f"**{text}**"

def bold_until_colon(text):
    first_colon_index = text.find(':')
    if first_colon_index != -1:
        return f"**{text[:first_colon_index + 1]}**{text[first_colon_index + 1:]}"
    else:
        # If there's no colon, bold the entire block
        return f"**{text}**"

def check_for_letters(text):
    return any(char.isalpha() for char in text)

def find_character_after_index(text, char, start_index):
    """
    Finds the first instance of a character in a string after a certain index.

    Args:
        text (str): The string to search in.
        char (str): The character to find.
        start_index (int): The index to start searching from.

    Returns:
        int: The index of the first instance of the character after the start index, or -1 if not found.
    """
    if start_index < 0 or start_index >= len(text):
        return -1

    for i in range(start_index, len(text)):
        if text[i] == char:
            return i
    return -1

def generate_mkdocs_nav(folder_path, output_file="mkdocs_nav.txt", header_name=None):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]

    # Prepare (display_name, filename) tuples
    entries = []
    for filename in md_files:
        name_without_ext = os.path.splitext(filename)[0]
        display_name = name_without_ext.replace("_", " ").capitalize()
        entries.append((display_name, filename))

    # Sort by display_name
    entries.sort(key=lambda x: x[0].lower())

    # Generate nav lines
    nav_lines = []
    if header_name:
        nav_lines.append(f"  - {header_name}")
    indent = "    " if header_name else ""

    for display_name, filename in entries:
        nav_lines.append(f"{indent}- {display_name}: {filename}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(nav_lines))

    print(f"Nav file written to: {output_file}")
# get plants with illustrations
images = pd.read_csv("./2_illustrations/assets.csv")

# get references to words in text
texts = pd.read_csv("./1_parse_text/verified_synonym_matches.csv")
# Create non-redundant df
import pandas as pd

# Assuming you already have `images` and `texts` as pandas DataFrames

# Ensure all spelling entries are lowercase for consistency
images["spelling"] = images["spelling"].str.lower()
texts["spelling"] = texts["spelling"].str.lower()

# Group by ID and official_name to get the text subchapters and type
text_grouped = texts.groupby(["ID", "official_name"]).agg({
    "subchapter": lambda x: "; ".join(sorted(set(x))),
    "spelling": lambda x: set(x),
    "type": lambda x: x.iloc[0]  # assumes one consistent type per ID
}).reset_index()

# Group by ID and official_name to get the illustration names and type
image_grouped = images.groupby(["ID", "official_name"]).agg({
    "illustration_name": lambda x: "; ".join(sorted(set(x))),
    "spelling": lambda x: set(x),
    "type": lambda x: x.iloc[0]  # assumes one consistent type per ID
}).reset_index()

# Merge the two grouped dataframes on ID and official_name
merged = pd.merge(text_grouped, image_grouped, on=["ID", "official_name"], how="outer", suffixes=("_text", "_image"))

# Replace NaNs with appropriate values
merged["subchapter"] = merged["subchapter"].fillna("NA")
merged["illustration_name"] = merged["illustration_name"].fillna("NA")

# Combine spellings from both sources
merged["spelling_combined"] = merged.apply(
    lambda row: (row["spelling_text"] if pd.notna(row["spelling_text"]) else set()) |
                (row["spelling_image"] if pd.notna(row["spelling_image"]) else set()),
    axis=1
)

# Determine type: prefer text type if available, else image type
merged["type"] = merged.apply(
    lambda row: row["type_text"] if pd.notna(row["type_text"]) else row["type_image"],
    axis=1
)

# Build the synonyms column
def get_synonyms(spellings, official_name):
    # Check if 'spellings' is a valid set, otherwise return "NA"
    if not isinstance(spellings, set):
        return "NA"

    official_lower = official_name.lower()
    # Filter out non-string values before processing
    filtered = {s for s in spellings if isinstance(s, str) and s != official_lower}
    
    if not filtered:
        return "NA"
    return "; ".join(sorted(filtered))

merged["synonyms"] = merged.apply(lambda row: get_synonyms(row["spelling_combined"], row["official_name"]), axis=1)

# Select and rename the final columns
final_df = merged[["ID", "official_name", "subchapter", "illustration_name", "synonyms", "type"]].rename(columns={
    "subchapter": "text_subchapters",
    "illustration_name": "illustrations"
})

# Save to CSV
final_df.to_csv("nahuatl_names_review.csv", index=False)
# Create different sections of pages
# get unique IDs
image_IDs = images["ID"].unique()
text_IDs = texts["ID"].unique()

# create a sections column, to know what section each ID is
sections = []
for i in range(len(texts)):
    curr_ID = texts["ID"].iloc[i]
    if curr_ID in image_IDs:
        sections.append("plant_image")
    else:
        sections.append(texts["type"].iloc[i])
texts["section"] = sections
# Create hyperlinked text files
# === Paths and setup ===
directory = "../data/texts"
directory_es = "../data/texts_ES"
output_folder = "5_hyperlinked_text"
os.makedirs(output_folder, exist_ok=True)

filenames = [f for f in os.listdir(directory) if f != ".DS_Store"]

for filename in filenames:
    english_path = os.path.join(directory, filename)
    spanish_path = os.path.join(directory_es, filename)

    # Extract metadata from filename
    pg_num = filename[(filename.index("_") + 2):(filename.index("_") + 5)].lstrip('0')
    link = f"https://archive.org/details/aztec-herbal-of-1552/page/{pg_num}"
    link_md = f"[{link}]({link})  \n"

    with open(english_path, 'r') as file:
        eng_text = file.read().replace('\n', '')
        eng_text = add_space_after_punctuation(eng_text)

    with open(spanish_path, 'r', encoding='utf-8') as file:
        spa_text = file.read().replace('\n', '')
        spa_text = add_space_after_punctuation(spa_text)

    # Extract synonyms from DataFrame for this subchapter
    if filename[0] != "c":
        subchap = filename[:3].lstrip('0')
        title = f"## Subchapter {subchap}  \n"
        texts["no_zero_subchap"] = texts["subchapter"].str.lstrip("0")
        
        # Filter out any non-string values from the synonyms list
        subchap_syn_raw = texts["spelling"][texts["no_zero_subchap"] == subchap].to_list()
        subchap_syn = [s for s in subchap_syn_raw if isinstance(s, str)]
        
        subchap_off = texts["official_name"][texts["no_zero_subchap"] == subchap].to_list()
    else:
        chap = filename[7:9].lstrip('0')
        title = f"## Chapter {chap}  \n"
        subchap_syn, subchap_off = [], []

    # === Replace and format English ===
    eng_output = replace_terms(eng_text, subchap_syn, subchap_off)
    eng_bold = bold_first_sentence(eng_output)
    eng_block = (
        '=== "English :flag_us:"\n'
        f"    {eng_bold}  \n"
        f"    {link_md}"
    )

    # === Replace and format Spanish ===
    spa_output = replace_terms(spa_text, subchap_syn, subchap_off)
    spa_bold = bold_first_sentence(spa_output)
    spa_block = (
        '=== "Español :flag_mx:"\n'
        f"    {spa_bold}  \n"
    )

    # === Combine and write ===
    final_md = title + '\n' + eng_block + '\n\n' + spa_block + '\n'

    output_file = os.path.join(output_folder, f"{subchap if filename[0] != 'c' else chap}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_md)
# Create subchapter and chapter markdown files
# get file names of all texts
directory = "./5_hyperlinked_text"
filenames = os.listdir(directory)

# remove .DS_Store if present
if '.DS_Store' in filenames:
    filenames.remove('.DS_Store')

# create a new folder to store chapter documents
folder_name = "6_chapters"  
os.makedirs(folder_name, exist_ok=True)

# get file names of all assets
asset_dir = "./3_assets"
asset_files = os.listdir(asset_dir)

# remove .DS_Store if present
if '.DS_Store' in asset_files:
    asset_files.remove('.DS_Store')

asset_chap_pgs = [] # save available pages with illustrations
asset_chap_inds = [] # save indices in asset files if page is available
for i in range(len(asset_files)):
    asset_id = asset_files[i][asset_files[i].index("_")+1:asset_files[i].index(".")]
    if asset_id[0]=="p":
        pg = asset_id[1:].lstrip("0")
        asset_chap_pgs.append(pg)
        asset_chap_inds.append(i)
    else:
        continue
for i in range(len(filenames)):
    file = filenames[i]
    id_num = file[:file.index(".")] # get chapter/subchapter number
    if not check_for_letters(id_num): # CHAPTERS
        path = directory+"/"+file
        with open(path, 'r') as file:
            text = file.read()
        path = folder_name+"/"+id_num +".md"  # change to your desired path
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    elif check_for_letters(id_num): # SUBCHAPTERS
        path = directory+"/"+file
        with open(path, 'r') as file:
            text = file.read()
        start_ind = text.rfind("/")+1 # USE LINK TO FIND PAGE NUMBER
        end_ind = find_character_after_index(text, ")", start_ind)
        page = text[start_ind:end_ind]
        if page in asset_chap_pgs:
            ill_ind = asset_chap_inds[asset_chap_pgs.index(page)] # get index of illustration
            ill_file = asset_files[ill_ind] # get illustration file name
            ill_path = "!["+ill_file+"](assets/"+ill_file+")  " # get text to add to markdown
    
            initials = ill_file[:ill_file.index("_")] # get initials
            attributions = [] # create a list of attributions
            for i in initials:
                if i=="A":
                    attributions.append("Leaf traces by: Alejandra Rougon-Cardoso, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  ")
                elif i=="D":
                    attributions.append("Leaf traces by: Daniel H. Chitwood, Michigan State University, USA  ")
                elif i=="J":
                    attributions.append("Leaf traces by: Jimena Jazmin Hurtado Olvera, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  ")
                elif i=="K":
                    attributions.append("Leaf traces by: Kylie DeViller, Jodrey School of Computer Science, Acadia University, Canada  ")
                elif i=="L":
                    attributions.append("Leaf traces by: Lachlann Simms, Acadia University, Canada  ")
                elif i=="M":
                    attributions.append("Leaf traces by: Mariana Jaired Ruíz Amaro, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  ")
                elif i=="N":
                    attributions.append("Leaf traces by: J. Noé García-Chávez, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  ")
                elif i=="Z":
                    attributions.append("Leaf traces by: Zoë Migicovsky, Acadia University, Canada  ")
            
            if attributions:
                attr = attributions[0] # keep to just one attribution for now
            else:
                attr = "" # Default to an empty string if no attributions are found

            # remove first line of text
            lines = text.splitlines(True)
            remaining_lines = lines[1:]
            text = "".join(remaining_lines)
            
            # markdown
            final_md = text + "\n" + ill_path + "\n" + attr
    
            path = folder_name+"/"+id_num+"_ill.md"  # change to your desired path, indicate illustrated
            with open(path, "w", encoding="utf-8") as f:
                f.write(final_md)
        else:
            path = folder_name+"/"+id_num+".md"  # change to your desired path, no illustration

            # remove first line of text
            lines = text.splitlines(True)
            remaining_lines = lines[1:]
            text = "".join(remaining_lines)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(text) # just text
        
# Plants with illustrations
# create a new folder to store chapter documents
folder_name = "7_plants_with_illlustrations"
os.makedirs(folder_name, exist_ok=True)
# for each ID with an illustration
for id in range(len(image_IDs)):

    curr_ID = image_IDs[id] # get the ID
    
    ###
    # ILLUSTRATIONS
    ###
    image_files = np.array(images["filename"][images["ID"]==curr_ID]) #.unique() # get associated files for image
    
    # image refs
    image_refs = [] # store references to images
    for img in image_files:
        image_ref = "!["+img+"](assets/"+img+")" # generate the markdown call for the image
        image_ref = image_ref+"  "+  "\n"
        image_refs.append(image_ref)
    
    # attributions
    
    attr_each_file = [] # get attributions for each of the files
    for j in range(len(image_files)):
        initials = image_files[j].split("_")[0] # get initials for attribution
        attributions = [] # create a list of attributions
        for i in initials:
            if i=="A":
                attributions.append("Leaf traces by: Alejandra Rougon-Cardoso, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  \n")
            elif i=="D":
                attributions.append("Leaf traces by: Daniel H. Chitwood, Michigan State University, USA  \n")
            elif i=="J":
                attributions.append("Leaf traces by: Jimena Jazmin Hurtado Olvera, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  \n")
            elif i=="K":
                attributions.append("Leaf traces by: Kylie DeViller, Jodrey School of Computer Science, Acadia University, Canada  \n")
            elif i=="L":
                attributions.append("Leaf traces by: Lachlann Simms, Acadia University, Canada  \n")
            elif i=="M":
                attributions.append("Leaf traces by: Mariana Jaired Ruíz Amaro, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  \n")
            elif i=="N":
                attributions.append("Leaf traces by: J. Noé García-Chávez, Laboratory of Agrigenomic Sciences, ENES Unidad León, México  \n")
            elif i=="Z":
                attributions.append("Leaf traces by: Zoë Migicovsky, Acadia University, Canada  \n")
        
        all_attr = "".join(attributions)
        attr_each_file.append(all_attr)
    
    ###
    # TEXTS
    ###
    
    # get unique subchapters for current ID
    subchapters = np.unique(texts["subchapter"][texts["ID"]==curr_ID].values)
    
    # for each subchapter, get associated hyperlinked texts
    curr_texts = []
    for s in range(len(subchapters)):
        path = "5_hyperlinked_text/"+subchapters[s].lstrip("0")+".txt"
        with open(path, 'r') as file:
            text = file.read()
        curr_texts.append(text)
    
    all_texts = "".join(curr_texts)
    
    ###
    # SYNONYMS
    ###

    # get official name
    
    if all_texts=="":
        name = images["official_name"][images["ID"]==curr_ID].unique()[0] # get official name from images if no text
    else:
        name = texts["official_name"][texts["ID"]==curr_ID].unique()[0] # get official name
    
    # FIX: Convert the column to string before finding unique values
    syns = np.unique(texts["spelling"][texts["ID"] == curr_ID].astype(str).str.lower().values)
    
    # Lowercase official name for comparison
    official_name_lc = name.lower()
    
    # Remove official name from list of variants
    syns = [s for s in syns if s != official_name_lc]
    
    # Build bullet list if any variants remain
    if len(syns) > 0:
        syn_lines = ["**Variants:**\n"]
        syn_lines += [f"- {syn}" for syn in syns]
        syn_text = "\n".join(syn_lines) + "\n\n"
    else:
        syn_text = ""

    
    ###
    # PUT TOGETHER AND SAVE
    ###
    
    # put it all together
    final_text = syn_text + all_texts
    
    for r in range(len(image_refs)):
        final_text = "".join([final_text,image_refs[r]])
        final_text = "".join([final_text,attr_each_file[r]])
    
    # save
    path = folder_name + "/" + name + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text) # just text

generate_mkdocs_nav(folder_name, output_file=folder_name+"_nav.txt", header_name="Plants with illustrations")
# Plants without illustration
# create a new folder to store chapter documents
folder_name = "8_plants"
os.makedirs(folder_name, exist_ok=True)

# find IDs for plants with no images
plants = texts["ID"][texts["section"]=="plant"].unique()

# for each plant with no illustration ID
for id in range(len(plants)):

    curr_ID = plants[id] # get the ID

    ###
    # TEXTS
    ###
    
    # get unique subchapters for current ID
    subchapters = np.unique(texts["subchapter"][texts["ID"]==curr_ID].values)
    
    # for each subchapter, get associated hyperlinked texts
    curr_texts = []
    for s in range(len(subchapters)):
        path = "5_hyperlinked_text/"+subchapters[s].lstrip("0")+".txt"
        with open(path, 'r') as file:
            text = file.read()
        curr_texts.append(text)
    
    all_texts = "".join(curr_texts)
    
    ###
    # SYNONYMS
    ###
    
    # get official name
    
    name = texts["official_name"][texts["ID"]==curr_ID].unique()[0] # get official name
    
    # FIX: Convert the column to string before finding unique values
    syns = np.unique(texts["spelling"][texts["ID"] == curr_ID].astype(str).str.lower().values)
    
    # Lowercase official name for comparison
    official_name_lc = name.lower()
    
    # Remove official name from list of variants
    syns = [s for s in syns if s != official_name_lc]
    
    # Build bullet list if any variants remain
    if len(syns) > 0:
        syn_lines = ["**Variants:**\n"]
        syn_lines += [f"- {syn}" for syn in syns]
        syn_text = "\n".join(syn_lines) + "\n\n"
    else:
        syn_text = ""
    
    ###
    # PUT TOGETHER AND SAVE
    ###
    
    # put it all together
    final_text = syn_text + all_texts
    
    # save
    path = folder_name + "/" + name + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text) # just text

generate_mkdocs_nav(folder_name, output_file=folder_name+"_nav.txt", header_name="Plants without illustrations")

# Stones
# create a new folder to store chapter documents
folder_name = "9_stones"  
os.makedirs(folder_name, exist_ok=True)

# find IDs for stones
stones = texts["ID"][texts["section"]=="stone"].unique()

# for each stone
for id in range(len(stones)):

    curr_ID = stones[id] # get the ID

    ###
    # TEXTS
    ###
    
    # get unique subchapters for current ID
    subchapters = np.unique(texts["subchapter"][texts["ID"]==curr_ID].values)
    
    # for each subchapter, get associated hyperlinked texts
    curr_texts = []
    for s in range(len(subchapters)):
        path = "5_hyperlinked_text/"+subchapters[s].lstrip("0")+".txt"
        with open(path, 'r') as file:
            text = file.read()
        curr_texts.append(text)
    
    all_texts = "".join(curr_texts)
    
    ###
    # SYNONYMS
    ###
    
    # get official name
    
    name = texts["official_name"][texts["ID"]==curr_ID].unique()[0] # get official name
    
    # FIX: Convert the column to string before finding unique values
    syns = np.unique(texts["spelling"][texts["ID"] == curr_ID].astype(str).str.lower().values)
    
    # Lowercase official name for comparison
    official_name_lc = name.lower()
    
    # Remove official name from list of variants
    syns = [s for s in syns if s != official_name_lc]
    
    # Build bullet list if any variants remain
    if len(syns) > 0:
        syn_lines = ["**Variants:**\n"]
        syn_lines += [f"- {syn}" for syn in syns]
        syn_text = "\n".join(syn_lines) + "\n\n"
    else:
        syn_text = ""
    
    ###
    # PUT TOGETHER AND SAVE
    ###
    
    # put it all together
    final_text = syn_text + all_texts
    
    # save
    path = folder_name + "/" + name + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text) # just text

generate_mkdocs_nav(folder_name, output_file=folder_name+"_nav.txt", header_name="Stones")

# Animals
# create a new folder to store chapter documents
folder_name = "10_animals"  
os.makedirs(folder_name, exist_ok=True)

# find IDs for animals
animals = texts["ID"][texts["section"]=="animal"].unique()

# for each animal
for id in range(len(animals)):

    curr_ID = animals[id] # get the ID

    ###
    # TEXTS
    ###
    
    # get unique subchapters for current ID
    subchapters = np.unique(texts["subchapter"][texts["ID"]==curr_ID].values)
    
    # for each subchapter, get associated hyperlinked texts
    curr_texts = []
    for s in range(len(subchapters)):
        path = "5_hyperlinked_text/"+subchapters[s].lstrip("0")+".txt"
        with open(path, 'r') as file:
            text = file.read()
        curr_texts.append(text)
    
    all_texts = "".join(curr_texts)
    
    ###
    # SYNONYMS
    ###
    
    # get official name
    
    name = texts["official_name"][texts["ID"]==curr_ID].unique()[0] # get official name
    
    # FIX: Convert the column to string before finding unique values
    syns = np.unique(texts["spelling"][texts["ID"] == curr_ID].astype(str).str.lower().values)
    
    # Lowercase official name for comparison
    official_name_lc = name.lower()
    
    # Remove official name from list of variants
    syns = [s for s in syns if s != official_name_lc]
    
    # Build bullet list if any variants remain
    if len(syns) > 0:
        syn_lines = ["**Variants:**\n"]
        syn_lines += [f"- {syn}" for syn in syns]
        syn_text = "\n".join(syn_lines) + "\n\n"
    else:
        syn_text = ""
    
    ###
    # PUT TOGETHER AND SAVE
    ###
    
    # put it all together
    final_text = syn_text + all_texts
    
    # save
    path = folder_name + "/" + name + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text) # just text

generate_mkdocs_nav(folder_name, output_file=folder_name+"_nav.txt", header_name="Animals")
# Birds
# create a new folder to store chapter documents
folder_name = "11_birds"  
os.makedirs(folder_name, exist_ok=True)

# find IDs for birds
birds = texts["ID"][texts["section"]=="bird"].unique()

# for each bird
for id in range(len(birds)):

    curr_ID = birds[id] # get the ID

    ###
    # TEXTS
###
    
    # get unique subchapters for current ID
    subchapters = np.unique(texts["subchapter"][texts["ID"]==curr_ID].values)
    
    # for each subchapter, get associated hyperlinked texts
    curr_texts = []
    for s in range(len(subchapters)):
        path = "5_hyperlinked_text/"+subchapters[s].lstrip("0")+".txt"
        with open(path, 'r') as file:
            text = file.read()
        curr_texts.append(text)
    
    all_texts = "".join(curr_texts)
    
    ###
    # SYNONYMS
    ###
    
    # get official name
    
    name = texts["official_name"][texts["ID"]==curr_ID].unique()[0] # get official name
    
    # FIX: Convert the column to string before finding unique values
    syns = np.unique(texts["spelling"][texts["ID"] == curr_ID].astype(str).str.lower().values)
    
    # Lowercase official name for comparison
    official_name_lc = name.lower()
    
    # Remove official name from list of variants
    syns = [s for s in syns if s != official_name_lc]
    
    # Build bullet list if any variants remain
    if len(syns) > 0:
        syn_lines = ["**Variants:**\n"]
        syn_lines += [f"- {syn}" for syn in syns]
        syn_text = "\n".join(syn_lines) + "\n\n"
    else:
        syn_text = ""
    
    ###
    # PUT TOGETHER AND SAVE
    ###
    
    # put it all together
    final_text = syn_text + all_texts
    
    # save
    path = folder_name + "/" + name + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text) # just text

generate_mkdocs_nav(folder_name, output_file=folder_name+"_nav.txt", header_name="Birds")
# Other
# create a new folder to store chapter documents
folder_name = "12_other"  
os.makedirs(folder_name, exist_ok=True)

# find IDs for other
other = texts["ID"][texts["section"]=="other"].unique()

# for each other
for id in range(len(other)):

    curr_ID = other[id] # get the ID

    ###
    # TEXTS
    ###
    
    # get unique subchapters for current ID
    subchapters = np.unique(texts["subchapter"][texts["ID"]==curr_ID].values)
    
    # for each subchapter, get associated hyperlinked texts
    curr_texts = []
    for s in range(len(subchapters)):
        path = "5_hyperlinked_text/"+subchapters[s].lstrip("0")+".txt"
        with open(path, 'r') as file:
            text = file.read()
        curr_texts.append(text)
    
    all_texts = "".join(curr_texts)
    
    ###
    # SYNONYMS
    ###
    
    # get official name
    
    name = texts["official_name"][texts["ID"]==curr_ID].unique()[0] # get official name
    
    # FIX: Convert the column to string before finding unique values
    syns = np.unique(texts["spelling"][texts["ID"] == curr_ID].astype(str).str.lower().values)
    
    # Lowercase official name for comparison
    official_name_lc = name.lower()
    
    # Remove official name from list of variants
    syns = [s for s in syns if s != official_name_lc]
    
    # Build bullet list if any variants remain
    if len(syns) > 0:
        syn_lines = ["**Variants:**\n"]
        syn_lines += [f"- {syn}" for syn in syns]
        syn_text = "\n".join(syn_lines) + "\n\n"
    else:
        syn_text = ""
    ###
    # PUT TOGETHER AND SAVE
    ###

    # put it all together
    final_text = syn_text + all_texts
    
    # save
    path = folder_name + "/" + name + ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text) # just text

generate_mkdocs_nav(folder_name, output_file=folder_name+"_nav.txt", header_name="Other")
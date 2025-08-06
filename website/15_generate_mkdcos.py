import os
import shutil
from pathlib import Path

# --- Configuration ---
WEBSITE_ROOT = Path(os.getcwd())
OUTPUT_DIR = WEBSITE_ROOT.parent / "output_site"
DOCS_DIR = OUTPUT_DIR / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

# --- Helper Functions ---
def read_and_format_nav_files(file_path, base_indent):
    """
    Reads a nav file, skips the first line (the section title), and
    returns the remaining content with a consistent indentation.
    """
    content = ""
    if file_path.exists():
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Skip the first line and process the rest
            for line in lines[1:]:
                stripped_line = line.strip()
                if stripped_line:
                    content += " " * base_indent + stripped_line + "\n"
    return content

# --- Main Logic ---
def create_mkdocs_site():
    """
    Assembles all generated files and assets into a complete MkDocs site
    ready for deployment.
    """
    print("Starting MkDocs site assembly...")
    
    # 1. Create the top-level output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Copy markdown files
    print("Copying markdown files...")
    md_files_source = WEBSITE_ROOT / "14_all_md_files"
    if not md_files_source.exists():
        print(f"Error: Directory not found at {md_files_source}")
        return
    shutil.copytree(md_files_source, DOCS_DIR, dirs_exist_ok=True)

    # 3. Copy main assets folder
    print("Copying primary assets...")
    assets_source = WEBSITE_ROOT / "3_assets"
    if not assets_source.exists():
        print(f"Error: Directory not found at {assets_source}")
        return
    shutil.copytree(assets_source, ASSETS_DIR, dirs_exist_ok=True)

    # 4. Copy individual website elements
    print("Copying specific website elements...")
    elements_source = WEBSITE_ROOT / "0_website_elements"
    if not elements_source.exists():
        print(f"Error: Directory not found at {elements_source}")
        return

    shutil.copy(elements_source / "icon.png", ASSETS_DIR / "icon.png")
    shutil.copy(elements_source / "opening_image.jpg", ASSETS_DIR / "opening_image.jpg")
    shutil.copy(elements_source / "index.md", DOCS_DIR / "index.md")

    # 5. Construct and write the mkdocs.yml file
    print("Generating mkdocs.yml file...")

    # Define the 'Welcome' and 'Chapters' nav blocks as a static string
    chapters_nav_block = """- Welcome | Bienvenid@ | Ximopanōltih:
    - Welcome | Bienvenid@ | Ximopanōltih: index.md
- Chapters:
    - Chapter 1:
      - "Opening": 1.md
      - "Subchapter 1a✎": 1a_ill.md
      - "Subchapter 1b✎": 1b_ill.md
      - "Subchapter 1c✎": 1c_ill.md
      - "Subchapter 1d✎": 1d_ill.md
      - "Subchapter 1e✎": 1e_ill.md
      - "Subchapter 1f✎": 1f_ill.md
    - Chapter 2:
      - "Opening": 2.md
      - "Subchapter 2a": 2a.md
      - "Subchapter 2b✎": 2b_ill.md
      - "Subchapter 2c": 2c.md
      - "Subchapter 2d✎": 2d_ill.md
      - "Subchapter 2e✎": 2e_ill.md
      - "Subchapter 2f✎": 2f_ill.md
      - "Subchapter 2g✎": 2g_ill.md
      - "Subchapter 2h": 2h.md
    - Chapter 3:
      - "Opening": 3.md
      - "Subchapter 3a✎": 3a_ill.md
    - Chapter 4:
      - "Opening": 4.md
      - "Subchapter 4a✎": 4a_ill.md
      - "Subchapter 4b✎": 4b_ill.md
      - "Subchapter 4c✎": 4c_ill.md
    - Chapter 5:
      - "Opening": 5.md
      - "Subchapter 5a": 5a.md
      - "Subchapter 5b": 5b.md
      - "Subchapter 5c✎": 5c_ill.md
      - "Subchapter 5d✎": 5d_ill.md
      - "Subchapter 5e✎": 5e_ill.md
      - "Subchapter 5f✎": 5f_ill.md
      - "Subchapter 5g✎": 5g_ill.md
      - "Subchapter 5h✎": 5h_ill.md
      - "Subchapter 5i✎": 5i_ill.md
    - Chapter 6:
      - "Opening": 6.md
      - "Subchapter 6a": 6a.md
      - "Subchapter 6b": 6b.md
      - "Subchapter 6c": 6c.md
      - "Subchapter 6d": 6d.md
      - "Subchapter 6e": 6e.md
      - "Subchapter 6f": 6f.md
      - "Subchapter 6g": 6g.md
      - "Subchapter 6h": 6h.md
      - "Subchapter 6i": 6i.md
    - Chapter 7:
      - "Opening": 7.md
      - "Subchapter 7a": 7a.md
      - "Subchapter 7b": 7b.md
      - "Subchapter 7c": 7c.md
      - "Subchapter 7d": 7d.md
      - "Subchapter 7e✎": 7e_ill.md
      - "Subchapter 7f✎": 7f_ill.md
      - "Subchapter 7g✎": 7g_ill.md
      - "Subchapter 7h✎": 7h_ill.md
      - "Subchapter 7i": 7i.md
      - "Subchapter 7j": 7j.md
      - "Subchapter 7k": 7k.md
      - "Subchapter 7l": 7l.md
      - "Subchapter 7m": 7m.md
      - "Subchapter 7n": 7n.md
    - Chapter 8:
      - "Opening": 8.md
      - "Subchapter 8a": 8a.md
      - "Subchapter 8b": 8b.md
      - "Subchapter 8c": 8c.md
      - "Subchapter 8d": 8d.md
      - "Subchapter 8e✎": 8e_ill.md
      - "Subchapter 8f✎": 8f_ill.md
      - "Subchapter 8g✎": 8g_ill.md
      - "Subchapter 8h✎": 8h_ill.md
      - "Subchapter 8i✎": 8i_ill.md
      - "Subchapter 8j✎": 8j_ill.md
      - "Subchapter 8k✎": 8k_ill.md
      - "Subchapter 8l": 8l.md
    - Chapter 9:
      - "Opening": 9.md
      - "Subchapter 9a✎": 9a_ill.md
      - "Subchapter 9b": 9b.md
      - "Subchapter 9c✎": 9c_ill.md
      - "Subchapter 9d✎": 9d_ill.md
      - "Subchapter 9e✎": 9e_ill.md
      - "Subchapter 9f✎": 9f_ill.md
      - "Subchapter 9g": 9g.md
      - "Subchapter 9h": 9h.md
      - "Subchapter 9i": 9i.md
      - "Subchapter 9j": 9j.md
      - "Subchapter 9k": 9k.md
      - "Subchapter 9l": 9l.md
      - "Subchapter 9m": 9m.md
      - "Subchapter 9n✎": 9n_ill.md
      - "Subchapter 9o✎": 9o_ill.md
      - "Subchapter 9p✎": 9p_ill.md
      - "Subchapter 9q✎": 9q_ill.md
    - Chapter 10:
      - "Opening": 10.md
      - "Subchapter 10a✎": 10a_ill.md
      - "Subchapter 10b✎": 10b_ill.md
      - "Subchapter 10c✎": 10c_ill.md
      - "Subchapter 10d✎": 10d_ill.md
      - "Subchapter 10e✎": 10e_ill.md
      - "Subchapter 10f✎": 10f_ill.md
      - "Subchapter 10g✎": 10g_ill.md
      - "Subchapter 10h": 10h.md
      - "Subchapter 10i✎": 10i_ill.md
      - "Subchapter 10j✎": 10j_ill.md
      - "Subchapter 10k✎": 10k_ill.md
    - Chapter 11:
      - "Opening": 11.md
      - "Subchapter 11a✎": 11a_ill.md
      - "Subchapter 11b✎": 11b_ill.md
      - "Subchapter 11c✎": 11c_ill.md
      - "Subchapter 11d✎": 11d_ill.md
      - "Subchapter 11e✎": 11e_ill.md
    - Chapter 12:
      - "Opening": 12.md
      - "Subchapter 12a✎": 12a_ill.md
      - "Subchapter 12b": 12b.md
    - Chapter 13:
      - "Opening": 13.md
      - "Subchapter 13a": 13a.md
"""

    # Read and format the content for each of the remaining categories,
    # hard-coding the top-level section headings to avoid the "404" problem.
    illustrations_content = read_and_format_nav_files(WEBSITE_ROOT / "7_plants_with_illlustrations_nav.txt", 4)
    plants_content = read_and_format_nav_files(WEBSITE_ROOT / "8_plants_nav.txt", 4)
    stones_content = read_and_format_nav_files(WEBSITE_ROOT / "9_stones_nav.txt", 4)
    animals_content = read_and_format_nav_files(WEBSITE_ROOT / "10_animals_nav.txt", 4)
    birds_content = read_and_format_nav_files(WEBSITE_ROOT / "11_birds_nav.txt", 4)
    other_content = read_and_format_nav_files(WEBSITE_ROOT / "12_other_nav.txt", 4)
    
    illustrations_nav_block = f"- Illustrations:\n{illustrations_content}"
    plants_nav_block = f"- Plants:\n{plants_content}"
    stones_nav_block = f"- Stones:\n{stones_content}"
    animals_nav_block = f"- Animals:\n{animals_content}"
    birds_nav_block = f"- Birds:\n{birds_content}"
    other_nav_block = f"- Other:\n{other_content}"

    # Construct the full mkdocs.yml content with all navigation blocks
    mkdocs_yml_content = f"""
# Project information
site_name: The De la Cruz-Badiano Nahuatl Herbal of 1552
site_url: https://danchitwood.github.io/DeLaCruzBadiano1552/
site_author: Dan Chitwood
site_description: Analysis of the De la Cruz-Badiano Nahuatl Herbal of 1552

# Repository
repo_name: DanChitwood/DeLaCruzBadiano1552/
repo_url: https://github.com/DanChitwood/DeLaCruzBadiano1552/

# Copyright
copyright: Copyright &copy; 2025 Dan Chitwood, Alejandra Rougon-Cardoso

# Configuration
theme:
  name: material
  features:
    #- toc.integrate
    - announce.dismiss
    #- content.action.edit
    #- content.action.view
    - content.code.annotate
    - content.code.copy
    # - content.code.select
    # - content.footnote.tooltips
    # - content.tabs.link
    - content.tooltips
    # - header.autohide
    # - navigation.expand
    - navigation.footer
    - navigation.indexes
    # - navigation.instant
    # - navigation.instant.prefetch
    # - navigation.instant.progress
    # - navigation.prune
    - navigation.sections
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.top
    - navigation.tracking
    - search.highlight
    - search.share
    - search.suggest
    - toc.follow
    # - toc.integrate

  palette:
    - media: "(prefers-color-scheme)"
      toggle:
        icon: material/link
        name: Switch to light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: pink
      accent: deep orange
      toggle:
        icon: material/toggle-switch
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: pink
      accent: deep orange
      toggle:
        icon: material/toggle-switch-off
        name: Switch to system preference

  font:
    text: Roboto
    code: Red Hat Mono
  logo: assets/icon.png
  favicon: assets/icon.png
  icon:
    repo: fontawesome/brands/git-alt

# Page tree
nav:
{chapters_nav_block}
{illustrations_nav_block}
{plants_nav_block}
{stones_nav_block}
{animals_nav_block}
{birds_nav_block}
{other_nav_block}

#plugins:
# - social

markdown_extensions:
  - attr_list
  - md_in_html
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details

extra:
  social:
    - icon: simple/youtube
      link: https://www.youtube.com/@DanChitwood
    - icon: simple/python
      link: https://danchitwood.github.io/plants_and_python/
"""
    
    # Write the complete mkdocs.yml file
    with open(OUTPUT_DIR / "mkdocs.yml", "w") as f:
        f.write(mkdocs_yml_content)

    print("\n✅ Site assembly complete! The files are in the 'output_site' directory.")
    print("To check your website locally, navigate to the `output_site` directory and run `mkdocs serve`.")

if __name__ == "__main__":
    create_mkdocs_site()
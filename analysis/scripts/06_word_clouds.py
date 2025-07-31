import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
import pickle
from wordcloud import WordCloud

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Input directories for data
INPUTS_MASTER_SHEET_PROCESSING_DIR = os.path.join(PROJECT_ROOT, "outputs", "master_sheet_processing")
INPUTS_TFIDF_VECTORIZATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "TF_IDF_vectorization")

# Output directory for figures
OUTPUTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

os.makedirs(OUTPUTS_FIGURES_DIR, exist_ok=True)

# Figure parameters
FIG_WIDTH = 8.5  # Total width of the figure in inches
FIG_HEIGHT = 11  # Total height of the figure in inches (for 6 rows of plots)
DPI = 300

# Number of clusters (must match the N_CLUSTERS used in 05_dimension_reduction_clustering.py)
N_CLUSTERS = 6 

print(f"--- Starting 06_word_clouds.py ---")

# --- Load Data ---
print(f"Loading aggregated texts, TF-IDF matrices, vocabularies, and cluster labels...")
try:
    df_texts = pd.read_csv(os.path.join(INPUTS_MASTER_SHEET_PROCESSING_DIR, "aggregated_plant_texts.csv"))
    
    tfidf_matrix_en = load_npz(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vectorization_matrix.npz'))
    tfidf_matrix_es = load_npz(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vectorization_matrix.npz'))
    
    with open(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vocab.pkl'), 'rb') as f:
        english_vocab = pickle.load(f)
    with open(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vocab.pkl'), 'rb') as f:
        spanish_vocab = pickle.load(f)
    
    nahuatl_cluster_df = pd.read_csv(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'nahuatl_cluster_labels.csv'))
    
    print("All necessary data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading required data: {e}")
    print("Please ensure 04_TF_IDF_vectorization.py and 05_dimension_reduction_clustering.py have been run successfully.")
    exit()

df_texts = df_texts.set_index('ID').loc[nahuatl_cluster_df['ID']].reset_index()
df_texts['aggregated_english_text'] = df_texts['aggregated_english_text'].fillna('')
df_texts['aggregated_spanish_text'] = df_texts['aggregated_spanish_text'].fillna('')


# --- Generate Word Clouds (Figure 2) ---
print("\nGenerating Figure 2: Word Clouds for each cluster (finalizing aesthetics)...")

cmap = plt.colormaps.get_cmap('Dark2')

# Removed fig.suptitle() as requested
fig, axes = plt.subplots(N_CLUSTERS, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

# Adjusted hspace to reduce vertical whitespace
plt.subplots_adjust(hspace=0.25, wspace=0.1) 

# Loop through each cluster
for i in range(N_CLUSTERS):
    # --- English Word Cloud ---
    english_rows_in_cluster = nahuatl_cluster_df[nahuatl_cluster_df['english_cluster'] == i].index.tolist()
    summed_tfidf_en = tfidf_matrix_en[english_rows_in_cluster, :].sum(axis=0).A1
    word_weights_en = {english_vocab[idx]: score for idx, score in enumerate(summed_tfidf_en) if score > 0}
    
    def color_func_en(word, font_size, position, orientation, random_state=None, **kwargs):
        rgb_color_255 = (np.array(cmap(i)[:3]) * 255).astype(int)
        return tuple(rgb_color_255)
    
    wc_en = WordCloud(background_color="white", width=800, height=400, 
                      max_words=100, contour_color='steelblue',
                      color_func=color_func_en, collocations=False,
                      random_state=42).generate_from_frequencies(word_weights_en)

    axes[i, 0].imshow(wc_en, interpolation='bilinear')
    # Reduced subtitle fontsize to 9
    axes[i, 0].set_title(f'English Cluster {i+1}', fontsize=9) 
    axes[i, 0].axis('off')

    # --- Spanish Word Cloud ---
    spanish_rows_in_cluster = nahuatl_cluster_df[nahuatl_cluster_df['spanish_cluster'] == i].index.tolist()
    summed_tfidf_es = tfidf_matrix_es[spanish_rows_in_cluster, :].sum(axis=0).A1
    word_weights_es = {spanish_vocab[idx]: score for idx, score in enumerate(summed_tfidf_es) if score > 0}

    def color_func_es(word, font_size, position, orientation, random_state=None, **kwargs):
        rgb_color_255 = (np.array(cmap(i)[:3]) * 255).astype(int)
        return tuple(rgb_color_255)
    
    wc_es = WordCloud(background_color="white", width=800, height=400,
                      max_words=100, contour_color='steelblue',
                      color_func=color_func_es, collocations=False,
                      random_state=42).generate_from_frequencies(word_weights_es)

    axes[i, 1].imshow(wc_es, interpolation='bilinear')
    # Reduced subtitle fontsize to 9
    axes[i, 1].set_title(f'Spanish Cluster {i+1}', fontsize=9) 
    axes[i, 1].axis('off')

# Save the figure
figure_filename = os.path.join(OUTPUTS_FIGURES_DIR, 'fig_wordcloud.png')
plt.savefig(figure_filename, dpi=DPI, bbox_inches='tight')
print(f"Figure 2 saved to: {figure_filename}")
plt.show()

print(f"--- 06_word_clouds.py Finished ---")
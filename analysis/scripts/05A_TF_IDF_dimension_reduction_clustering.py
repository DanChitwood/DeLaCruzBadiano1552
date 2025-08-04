import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import pickle
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import phate
import matplotlib.gridspec as gridspec

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUTS_MASTER_SHEET_PROCESSING_DIR = os.path.join(PROJECT_ROOT, "outputs", "master_sheet_processing")
INPUTS_TFIDF_VECTORIZATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "TF_IDF_vectorization")
OUTPUTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUTS_CLUSTERING_DATA_DIR = os.path.join(PROJECT_ROOT, "outputs", "clustering_data")

os.makedirs(OUTPUTS_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_CLUSTERING_DATA_DIR, exist_ok=True)

# Figure and Clustering Parameters
FIG_WIDTH = 8.5    # Total width of the figure in inches
FIG_HEIGHT = 11    # Total height of the figure in inches
DPI = 300
RANDOM_STATE = 42

# You can choose one of the following for the embedding method:
# 'PHATE', 'UMAP', 't-SNE'
EMBEDDING_METHOD = 'PHATE'
N_CLUSTERS = 5 # Number of clusters for K-Means

print(f"--- Starting final combined script ---")
print(f"Embedding Method: {EMBEDDING_METHOD}, Number of Clusters: {N_CLUSTERS}")

# --- Load Data ---
print(f"\nLoading all necessary data files...")
try:
    tfidf_matrix_en = load_npz(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vectorization_matrix.npz'))
    tfidf_matrix_es = load_npz(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vectorization_matrix.npz'))
    with open(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'nahuatl_ids.pkl'), 'rb') as f:
        nahuatl_ids = pickle.load(f)
    with open(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vocab.pkl'), 'rb') as f:
        english_vocab = pickle.load(f)
    with open(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vocab.pkl'), 'rb') as f:
        spanish_vocab = pickle.load(f)
    
    print("TF-IDF matrices, vocabularies, and Nahuatl IDs loaded.")

except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e}")
    print("Please ensure 04_TF_IDF_vectorization.py has been run successfully.")
    exit()


# --- Dimensionality Reduction based on user choice ---
print(f"\nPerforming {EMBEDDING_METHOD} dimension reduction...")

if EMBEDDING_METHOD == 'PHATE':
    phate_operator = phate.PHATE(n_components=2, random_state=RANDOM_STATE)
    coords_en = phate_operator.fit_transform(tfidf_matrix_en)
    coords_es = phate_operator.fit_transform(tfidf_matrix_es)
elif EMBEDDING_METHOD == 'UMAP':
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_STATE)
    coords_en = umap_reducer.fit_transform(tfidf_matrix_en)
    coords_es = umap_reducer.fit_transform(tfidf_matrix_es)
elif EMBEDDING_METHOD == 't-SNE':
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2500, random_state=RANDOM_STATE, learning_rate='auto', init='random')
    coords_en = tsne.fit_transform(tfidf_matrix_en)
    coords_es = tsne.fit_transform(tfidf_matrix_es)
else:
    print(f"Error: Invalid EMBEDDING_METHOD '{EMBEDDING_METHOD}'. Please choose from 'PHATE', 'UMAP', or 't-SNE'.")
    exit()


# --- K-Means Clustering on the Embedded Space ---
print(f"\nPerforming K-Means clustering with k={N_CLUSTERS} on {EMBEDDING_METHOD} embeddings...")
kmeans_en = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=10, random_state=RANDOM_STATE)
clusters_en = kmeans_en.fit_predict(coords_en)

kmeans_es = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=10, random_state=RANDOM_STATE)
clusters_es = kmeans_es.fit_predict(coords_es)

print("Clustering completed for both languages.")


# --- Create and Save Cluster Data ---
cluster_data_df = pd.DataFrame({
    'ID': nahuatl_ids,
    'english_cluster': clusters_en,
    'spanish_cluster': clusters_es
})
cluster_labels_path = os.path.join(OUTPUTS_CLUSTERING_DATA_DIR, 'nahuatl_cluster_labels.csv')
cluster_data_df.to_csv(cluster_labels_path, index=False)
print(f"Cluster labels saved to: {cluster_labels_path}")


# --- Generate a Combined Figure with Final Layout ---
print("\nGenerating final combined figure...")

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.15, wspace=0.05,
                       height_ratios=[1, 1, 1, 1])

# Use the specific hex codes provided
cluster_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02']


# --- English Plots (Rows 1 & 2) ---
# Row 1, Col 1: English Embedding Plot
ax0_0 = fig.add_subplot(gs[0, 0])
ax0_0.grid(True, linestyle='--', alpha=0.6)
ax0_0.set_xlabel('Dimension 1')
ax0_0.set_ylabel('Dimension 2')
for i in range(N_CLUSTERS):
    cluster_points = coords_en[clusters_en == i]
    ax0_0.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[cluster_colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)
ax0_0.legend(loc='best', frameon=False, fontsize='small')

# English Word Clouds
wc_axes_en = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
              fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
              fig.add_subplot(gs[1, 2])]

for i in range(N_CLUSTERS):
    ax = wc_axes_en[i]
    english_rows_in_cluster = np.where(clusters_en == i)[0]
    summed_tfidf_en = tfidf_matrix_en[english_rows_in_cluster, :].sum(axis=0).A1
    word_weights_en = {english_vocab[idx]: score for idx, score in enumerate(summed_tfidf_en) if score > 0}
    
    def color_func_en(word, font_size, position, orientation, random_state=None, **kwargs):
        return cluster_colors[i]
    
    wc_en = WordCloud(background_color="white", width=800, height=400, max_words=100,
                      color_func=color_func_en, collocations=False,
                      random_state=RANDOM_STATE).generate_from_frequencies(word_weights_en)

    ax.imshow(wc_en, interpolation='bilinear')
    ax.axis('off')


# --- Spanish Plots (Rows 3 & 4) ---
# Row 3, Col 1: Spanish Embedding Plot
ax2_0 = fig.add_subplot(gs[2, 0])
ax2_0.grid(True, linestyle='--', alpha=0.6)
ax2_0.set_xlabel('Dimension 1')
ax2_0.set_ylabel('Dimension 2')
for i in range(N_CLUSTERS):
    cluster_points = coords_es[clusters_es == i]
    ax2_0.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[cluster_colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)
ax2_0.legend(loc='best', frameon=False, fontsize='small')


# Spanish Word Clouds
wc_axes_es = [fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2]),
              fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
              fig.add_subplot(gs[3, 2])]

for i in range(N_CLUSTERS):
    ax = wc_axes_es[i]
    spanish_rows_in_cluster = np.where(clusters_es == i)[0]
    summed_tfidf_es = tfidf_matrix_es[spanish_rows_in_cluster, :].sum(axis=0).A1
    word_weights_es = {spanish_vocab[idx]: score for idx, score in enumerate(summed_tfidf_es) if score > 0}

    def color_func_es(word, font_size, position, orientation, random_state=None, **kwargs):
        return cluster_colors[i]
    
    wc_es = WordCloud(background_color="white", width=800, height=400, max_words=100,
                      color_func=color_func_es, collocations=False,
                      random_state=RANDOM_STATE).generate_from_frequencies(word_weights_es)

    ax.imshow(wc_es, interpolation='bilinear')
    ax.axis('off')

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
figure_filename = os.path.join(OUTPUTS_FIGURES_DIR, f'fig_combined_clustering_and_wordclouds.png')
plt.savefig(figure_filename, dpi=DPI, bbox_inches='tight')
plt.close(fig)

print(f"\nFinal combined figure saved to: {figure_filename}")
print(f"--- Script Finished ---")
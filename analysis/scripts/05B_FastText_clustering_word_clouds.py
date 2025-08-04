import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import phate
import matplotlib.gridspec as gridspec
from collections import defaultdict

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Input Directories
INPUTS_PROCESSING_DIR = os.path.join(PROJECT_ROOT, "outputs", "master_sheet_processing")
OUTPUTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUTS_CLUSTERING_DATA_DIR = os.path.join(PROJECT_ROOT, "outputs", "clustering_data")

os.makedirs(OUTPUTS_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_CLUSTERING_DATA_DIR, exist_ok=True)

# Figure and Clustering Parameters
FIG_WIDTH = 8.5    # Total width of the figure in inches
FIG_HEIGHT = 11    # Total height of the figure in inches
DPI = 300
RANDOM_STATE = 42

# Using UMAP as the default method
EMBEDDING_METHOD = 'UMAP'
N_CLUSTERS = 5 # Number of clusters for K-Means

print(f"--- Starting final combined script ---")
print(f"Embedding Method: {EMBEDDING_METHOD}, Number of Clusters: {N_CLUSTERS}")

# --- Load Data ---
print(f"\nLoading all necessary data files...")
try:
    fasttext_embeddings_en = np.load(os.path.join(INPUTS_PROCESSING_DIR, 'english_text_embeddings.npy'))
    fasttext_embeddings_es = np.load(os.path.join(INPUTS_PROCESSING_DIR, 'spanish_text_embeddings.npy'))
    aggregated_df = pd.read_csv(os.path.join(INPUTS_PROCESSING_DIR, 'aggregated_plant_texts.csv'))
    nahuatl_ids = aggregated_df['ID'].values.tolist()
    
    print("FastText embeddings and aggregated text data loaded.")

except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e}")
    print("Please ensure the FastText vectorization script has been run successfully.")
    exit()

# Reshape the FastText embeddings for clustering (from 3D to 2D)
embeddings_en = fasttext_embeddings_en.mean(axis=1)
embeddings_es = fasttext_embeddings_es.mean(axis=1)
print(f"Reshaped English embeddings shape: {embeddings_en.shape}")
print(f"Reshaped Spanish embeddings shape: {embeddings_es.shape}")

# --- Dimensionality Reduction based on user choice ---
print(f"\nPerforming {EMBEDDING_METHOD} dimension reduction...")

if EMBEDDING_METHOD == 'PHATE':
    phate_operator = phate.PHATE(n_components=2, random_state=RANDOM_STATE)
    coords_en = phate_operator.fit_transform(embeddings_en)
    coords_es = phate_operator.fit_transform(embeddings_es)
elif EMBEDDING_METHOD == 'UMAP':
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_STATE)
    coords_en = umap_reducer.fit_transform(embeddings_en)
    coords_es = umap_reducer.fit_transform(embeddings_es)
elif EMBEDDING_METHOD == 't-SNE':
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2500, random_state=RANDOM_STATE, learning_rate='auto', init='random')
    coords_en = tsne.fit_transform(embeddings_en)
    coords_es = tsne.fit_transform(embeddings_es)
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

cluster_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02']
custom_stopwords = STOPWORDS.union({'v'})

# --- NEW LOGIC: Calculate TF-ICF scores for word clouds ---
# Function to get TF-ICF scores for a given language
def get_tf_icf_scores(clusters, language_column, n_clusters, custom_stopwords):
    cluster_texts = [aggregated_df[clusters == i][language_column].fillna('').str.cat(sep=' ') for i in range(n_clusters)]
    
    # Calculate word frequencies per cluster
    cluster_word_freqs = []
    global_word_set = set()
    for text in cluster_texts:
        word_freqs = defaultdict(int)
        for word in text.split():
            # Basic cleaning
            word = word.lower().strip('.,;()[]')
            if word and word not in custom_stopwords:
                word_freqs[word] += 1
                global_word_set.add(word)
        cluster_word_freqs.append(word_freqs)
    
    # Calculate Inverse Cluster Frequency (ICF)
    icf = {}
    total_clusters = len(cluster_texts)
    for word in global_word_set:
        clusters_with_word = sum(1 for freqs in cluster_word_freqs if word in freqs)
        icf[word] = np.log(total_clusters / (clusters_with_word + 1))
        
    # Calculate TF-ICF score for each word in each cluster
    tf_icf_scores = []
    for i in range(n_clusters):
        scores = {}
        for word, freq in cluster_word_freqs[i].items():
            tf = freq
            scores[word] = tf * icf.get(word, 0)
        tf_icf_scores.append(scores)
        
    return tf_icf_scores

# Generate TF-ICF scores for both languages
tf_icf_en = get_tf_icf_scores(clusters_en, 'aggregated_english_text', N_CLUSTERS, custom_stopwords)
tf_icf_es = get_tf_icf_scores(clusters_es, 'aggregated_spanish_text', N_CLUSTERS, custom_stopwords)


# --- English Plots (Rows 1 & 2) ---
# Row 1, Col 1: English Embedding Plot
ax0_0 = fig.add_subplot(gs[0, 0])
ax0_0.grid(True, linestyle='--', alpha=0.6)
ax0_0.set_xlabel('Dimension 1')
ax0_0.set_ylabel('Dimension 2')
for i in range(N_CLUSTERS):
    cluster_points = coords_en[clusters_en == i]
    ax0_0.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[cluster_colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)

# English Word Clouds
wc_axes_en = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
              fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
              fig.add_subplot(gs[1, 2])]

for i in range(N_CLUSTERS):
    ax = wc_axes_en[i]
    def color_func_en(word, font_size, position, orientation, random_state=None, **kwargs):
        return cluster_colors[i]
    
    wc_en = WordCloud(background_color="white", width=800, height=400, max_words=100,
                      color_func=color_func_en, collocations=False,
                      random_state=RANDOM_STATE).generate_from_frequencies(tf_icf_en[i])

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

# Spanish Word Clouds
wc_axes_es = [fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2]),
              fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
              fig.add_subplot(gs[3, 2])]

for i in range(N_CLUSTERS):
    ax = wc_axes_es[i]
    def color_func_es(word, font_size, position, orientation, random_state=None, **kwargs):
        return cluster_colors[i]
    
    wc_es = WordCloud(background_color="white", width=800, height=400, max_words=100,
                      color_func=color_func_es, collocations=False,
                      random_state=RANDOM_STATE).generate_from_frequencies(tf_icf_es[i])

    ax.imshow(wc_es, interpolation='bilinear')
    ax.axis('off')

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
figure_filename = os.path.join(OUTPUTS_FIGURES_DIR, f'fig_combined_clustering_and_wordclouds_fasttext_umap_tficf.png')
plt.savefig(figure_filename, dpi=DPI, bbox_inches='tight')
plt.close(fig)

print(f"\nFinal combined figure saved to: {figure_filename}")
print(f"--- Script Finished ---")
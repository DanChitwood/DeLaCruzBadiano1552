import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
import pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUTS_TFIDF_VECTORIZATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "TF_IDF_vectorization")
OUTPUTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

os.makedirs(OUTPUTS_FIGURES_DIR, exist_ok=True)

# Figure parameters
FIG_WIDTH = 8.5   # Total width of the figure in inches
FIG_HEIGHT = 4.25 # Adjusted height to maintain aspect ratio for 2 side-by-side plots
DPI = 300
TSNE_RANDOM_STATE = 42
KMEANS_RANDOM_STATE = 42

print(f"--- Starting 05_dimension_reduction_clustering.py ---")

# --- Load Data ---
print(f"Loading TF-IDF matrices and Nahuatl IDs from: {INPUTS_TFIDF_VECTORIZATION_DIR}")
try:
    tfidf_matrix_en = load_npz(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'english_TF_IDF_vectorization_matrix.npz'))
    tfidf_matrix_es = load_npz(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'spanish_TF_IDF_vectorization_matrix.npz'))
    with open(os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'nahuatl_ids.pkl'), 'rb') as f:
        nahuatl_ids = pickle.load(f)
    
    print(f"Loaded English TF-IDF matrix shape: {tfidf_matrix_en.shape}")
    print(f"Loaded Spanish TF-IDF matrix shape: {tfidf_matrix_es.shape}")
    print(f"Loaded {len(nahuatl_ids)} Nahuatl IDs.")

except FileNotFoundError:
    print(f"Error: Required TF-IDF data files not found in {INPUTS_TFIDF_VECTORIZATION_DIR}.")
    print("Please ensure 04_TF_IDF_vectorization.py has been run successfully and saved outputs to this directory.")
    exit()

# --- Estimate optimal K using Elbow Method (on English data as representative) ---
print("\n--- Estimating optimal 'k' for K-Means clustering using the Elbow Method (on English data) ---")
wcss = [] 
K_range = range(2, 16) 
for k in K_range:
    kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=KMEANS_RANDOM_STATE)
    kmeans_model.fit(tfidf_matrix_en)
    wcss.append(kmeans_model.inertia_)
    print(f"  k={k}: Inertia={kmeans_model.inertia_:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K (English TF-IDF Data)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.grid(True)
elbow_plot_filename = os.path.join(OUTPUTS_FIGURES_DIR, 'elbow_method_k_estimation.png')
plt.savefig(elbow_plot_filename, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Elbow method plot saved to: {elbow_plot_filename}")
print(f"Please inspect '{elbow_plot_filename}' to help determine optimal N_CLUSTERS.")

# --- Define Number of Clusters ---
N_CLUSTERS = 6 
print(f"\nProceeding with N_CLUSTERS = {N_CLUSTERS} for K-Means clustering.")


# --- K-Means Clustering ---
print(f"\nPerforming K-Means clustering with k={N_CLUSTERS} for English text...")
kmeans_en = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=10, random_state=KMEANS_RANDOM_STATE)
clusters_en = kmeans_en.fit_predict(tfidf_matrix_en)

print(f"Performing K-Means clustering with k={N_CLUSTERS} for Spanish text...")
kmeans_es = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=10, random_state=KMEANS_RANDOM_STATE)
clusters_es = kmeans_es.fit_predict(tfidf_matrix_es)

print("K-Means clustering completed for both languages.")

# --- t-SNE Dimension Reduction ---
print("\nPerforming t-SNE dimension reduction for English data (this may take some time)...")
tsne_en = TSNE(n_components=2, perplexity=30, n_iter=2500, random_state=TSNE_RANDOM_STATE, learning_rate='auto', init='random')
tsne_coords_en = tsne_en.fit_transform(tfidf_matrix_en)

print("Performing t-SNE dimension reduction for Spanish data (this may take some time)...")
tsne_es = TSNE(n_components=2, perplexity=30, n_iter=2500, random_state=TSNE_RANDOM_STATE, learning_rate='auto', init='random')
tsne_coords_es = tsne_es.fit_transform(tfidf_matrix_es)

print("t-SNE dimension reduction completed for both languages.")

# --- Visualization (Figure 1) ---
print("\nGenerating Figure 1: t-SNE plots with K-Means clusters (finalizing aesthetics)...")

cmap = plt.cm.get_cmap('Dark2', N_CLUSTERS)

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

# Plot English t-SNE
for i in range(N_CLUSTERS):
    cluster_points = tsne_coords_en[clusters_en == i]
    axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[cmap(i)], label=f'{i+1}', s=20, alpha=0.7) # Label changed to just number
axes[0].set_title('English')
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize='small') # frameon=False, fontsize='small'


# Plot Spanish t-SNE
for i in range(N_CLUSTERS):
    cluster_points = tsne_coords_es[clusters_es == i]
    axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[cmap(i)], label=f'{i+1}', s=20, alpha=0.7) # Label changed to just number
axes[1].set_title('Spanish') # Changed title to "Spanish"
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize='small') # frameon=False, fontsize='small'

plt.tight_layout()
figure_filename = os.path.join(OUTPUTS_FIGURES_DIR, 'fig_TFIDF_tsne_clusters.png')
plt.savefig(figure_filename, dpi=DPI, bbox_inches='tight')
print(f"Figure 1 saved to: {figure_filename}")
plt.show()


# --- Save Cluster Labels ---
print("\nSaving cluster labels for both languages...")
cluster_data_df = pd.DataFrame({
    'ID': nahuatl_ids,
    'english_cluster': clusters_en,
    'spanish_cluster': clusters_es
})
cluster_labels_path = os.path.join(INPUTS_TFIDF_VECTORIZATION_DIR, 'nahuatl_cluster_labels.csv')
cluster_data_df.to_csv(cluster_labels_path, index=False)

print(f"Cluster labels saved to: {cluster_labels_path}")

# --- Print Cluster Distribution ---
print("\n--- Cluster Distribution (English) ---")
print(pd.Series(clusters_en).value_counts().sort_index())
print("\n--- Cluster Distribution (Spanish) ---")
print(pd.Series(clusters_es).value_counts().sort_index())

print(f"--- 05_dimension_reduction_clustering.py Finished ---")
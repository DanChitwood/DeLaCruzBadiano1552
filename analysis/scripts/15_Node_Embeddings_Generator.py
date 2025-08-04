import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import os
import pickle
from collections import defaultdict

# --- Configuration (Adjusted for script location and data paths) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

OUTPUTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUTS_PROCESSING_DIR = os.path.join(OUTPUTS_BASE_DIR, "master_sheet_processing")
VERIFIED_LINKS_CSV = os.path.join(OUTPUTS_PROCESSING_DIR, "verified_subchapter_links.csv")

# Output directory for the embeddings
EMBEDDINGS_DIR = os.path.join(OUTPUTS_BASE_DIR, "three_tower_embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
NODE_EMBEDDINGS_PKL = os.path.join(EMBEDDINGS_DIR, "nahuatl_node_embeddings.pkl")

# Node2Vec parameters
EMBEDDING_SIZE = 128  # Must match the embedding size of the other towers
WALK_LENGTH = 30
NUM_WALKS = 200
P = 1.0  # Return hyperparameter, controls DFS vs BFS
Q = 1.0  # In-out hyperparameter, controls DFS vs BFS

print(f"Loading verified links from: {VERIFIED_LINKS_CSV}")
try:
    verified_df = pd.read_csv(VERIFIED_LINKS_CSV)
    print(f"Loaded {len(verified_df)} verified links records.")
except FileNotFoundError:
    print(f"Error: The file '{VERIFIED_LINKS_CSV}' was not found.")
    print("Please ensure the 'verified_subchapter_links.csv' is in the correct location.")
    exit()

# Filter for English language data as requested, even though official_name is Nahuatl
english_df = verified_df[verified_df['language'] == 'english'].copy()
print(f"Filtered to {len(english_df)} English language records.")

# Drop rows with missing values that are critical for graph building
english_df.dropna(subset=['official_name', 'subchapter_code'], inplace=True)
english_df = english_df[english_df['official_name'].astype(str).str.strip() != '']
english_df = english_df[english_df['subchapter_code'].astype(str).str.strip() != '']

if english_df.empty:
    print("No valid English entries found after filtering and cleaning. Cannot build graph.")
    exit()

# Dictionary to store co-occurrence counts of official_names
co_occurrence_counts = defaultdict(int)

# Group by subchapter to find co-occurring official_names
subchapter_to_names = defaultdict(list)
for index, row in english_df.iterrows():
    subchapter_to_names[str(row['subchapter_code'])].append(str(row['official_name']).strip().lower())

# Iterate through each subchapter and count co-occurrences
for subchapter_code, names_list in subchapter_to_names.items():
    unique_names_in_subchapter = sorted(list(set(names_list)))
    
    for i in range(len(unique_names_in_subchapter)):
        for j in range(i + 1, len(unique_names_in_subchapter)):
            name1 = unique_names_in_subchapter[i]
            name2 = unique_names_in_subchapter[j]
            edge = tuple(sorted((name1, name2)))
            co_occurrence_counts[edge] += 1

# Create the graph
G = nx.Graph()

# Add nodes and edges based on co-occurrence counts
all_names = set(english_df['official_name'].str.strip().str.lower().unique())
G.add_nodes_from(all_names)

for (name1, name2), weight in co_occurrence_counts.items():
    G.add_edge(name1, name2, weight=weight)

print(f"\nCreated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# --- Generate Node Embeddings using Node2Vec ---
# We use the full graph to generate rich embeddings, even if some nodes have low connectivity.
print("Generating Node2Vec embeddings...")
node2vec = Node2Vec(G, dimensions=EMBEDDING_SIZE, walk_length=WALK_LENGTH, 
                    num_walks=NUM_WALKS, p=P, q=Q, workers=1) 
# Note: For multiprocessing, increase 'workers' to number of CPU cores.
#       This is set to 1 for simplicity and reproducibility.
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Extract embeddings and save them to a dictionary
node_embeddings = {node: model.wv[node] for node in G.nodes()}

# Save the embeddings to a pickle file
with open(NODE_EMBEDDINGS_PKL, 'wb') as f:
    pickle.dump(node_embeddings, f)

print(f"\nNode embeddings generated for {len(node_embeddings)} nodes.")
print(f"Embeddings saved to: {NODE_EMBEDDINGS_PKL}")
print("Script completed successfully.")
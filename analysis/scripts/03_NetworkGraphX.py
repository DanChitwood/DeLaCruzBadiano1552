import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np # Import numpy for setting the random seed

# --- Configuration (Adjusted for script location and data paths) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

OUTPUTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUTS_PROCESSING_DIR = os.path.join(OUTPUTS_BASE_DIR, "master_sheet_processing")
VERIFIED_LINKS_CSV = os.path.join(OUTPUTS_PROCESSING_DIR, "verified_subchapter_links.csv")

# Ensure output directory exists for plots
FIGURES_DIR = os.path.join(OUTPUTS_BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Loading verified links from: {VERIFIED_LINKS_CSV}")
try:
    verified_df = pd.read_csv(VERIFIED_LINKS_CSV)
    print(f"Loaded {len(verified_df)} verified links records.")
except FileNotFoundError:
    print(f"Error: The file '{VERIFIED_LINKS_CSV}' was not found.")
    print("Please ensure the 'verified_subchapter_links.csv' is in the correct location.")
    print("Expected path (relative to script): ../outputs/master_sheet_processing/verified_subchapter_links.csv")
    exit() # Exit if the crucial file isn't found

# --- Prepare data for NetworkX graph ---

# 1. Filter for English language data as requested
english_df = verified_df[verified_df['language'] == 'english'].copy()
print(f"Filtered to {len(english_df)} English language records.")

# Drop rows where 'official_name' or 'subchapter_code' might be missing or empty
english_df.dropna(subset=['official_name', 'subchapter_code', 'type'], inplace=True)
english_df = english_df[english_df['official_name'].astype(str).str.strip() != '']
english_df = english_df[english_df['subchapter_code'].astype(str).str.strip() != '']

if english_df.empty:
    print("No valid English entries found after filtering and cleaning. Cannot build graph.")
    exit()

# Dictionary to store co-occurrence counts of official_names
co_occurrence_counts = defaultdict(int)

# Dictionary to map official_name to its primary plant type for coloring
official_name_to_type = {}
for index, row in english_df.iterrows():
    official_name_to_type[str(row['official_name']).strip().lower()] = str(row['type']).strip().lower()

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

# Define your custom type palette
type_pal = {"word":"k", # black - Assuming 'word' might be a placeholder or generic type
            "plant":"#1b9e77", # green
            "animal":"#d95f02", # orange
            "bird":"#7570b3", # purple
            "stone":"#e7298a", # magenta
            "other":"brown", # brown
           }

# Add nodes (unique official_names) and assign colors based on type_pal
actual_unique_types_in_data = set() 
for name in official_name_to_type.keys():
    plant_type = official_name_to_type.get(name, 'unknown') 
    actual_unique_types_in_data.add(plant_type) 
    G.add_node(name, type=plant_type, color=type_pal.get(plant_type, 'lightgray')) 

# Add edges based on co-occurrence counts
for (name1, name2), weight in co_occurrence_counts.items():
    G.add_edge(name1, name2, weight=weight)

print(f"\nCreated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Unique Official Names (nodes): {G.number_of_nodes()}")
print(f"Actual Unique Types in Data: {sorted(list(actual_unique_types_in_data))}")

# --- Visualization Parameters ---
# --- IMPORTANT: Adjust these parameters to improve visualization ---
FIG_WIDTH = 8.5 # inches
FIG_HEIGHT = 11 # inches

MIN_DEGREE_THRESHOLD = 10 # Threshold number of connections

# Node Sizing Parameters for dynamic scaling
MIN_NODE_SIZE = 10 # Minimum size for nodes, regardless of degree
DEGREE_SCALE_FACTOR = 10 # Multiplier for node size based on its degree

# Font Sizing Parameters for dynamic scaling
MIN_FONT_SIZE = 5 # Minimum font size for labels
FONT_DEGREE_SCALE_FACTOR = 0.05 # Multiplier for font size based on degree (adjust carefully!)

ALPHA_NODES = 0.6 
ALPHA_EDGES = 0.3 
EDGE_COLOR = "gray" 
EDGE_WIDTH_SCALE = 0.7 # Increased scale for edge width to make differences more dynamic

RANDOM_SEED = 42 # Set this to any integer for reproducible layouts

# --- Visualization ---

# Filter nodes based on MIN_DEGREE_THRESHOLD
nodes_to_draw = [node for node, degree in G.degree() if degree >= MIN_DEGREE_THRESHOLD]
subgraph = G.subgraph(nodes_to_draw)

print(f"\nDrawing subgraph with {subgraph.number_of_nodes()} nodes (min_degree={MIN_DEGREE_THRESHOLD}).")

if subgraph.number_of_nodes() == 0:
    print(f"No nodes to draw after filtering with min_degree={MIN_DEGREE_THRESHOLD}. Cannot generate graph.")
else:
    # Set the numpy random seed for reproducibility of layout algorithms
    np.random.seed(RANDOM_SEED)

    # Choose graph layout: Using Kamada-Kawai with scale=2
    pos = nx.kamada_kawai_layout(subgraph, scale=2) 
    
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT)) 
    ax = plt.gca() # Get the current axes to draw text labels individually

    # Calculate dynamic node sizes
    node_degrees = dict(subgraph.degree())
    node_sizes = [MIN_NODE_SIZE + node_degrees[node] * DEGREE_SCALE_FACTOR for node in subgraph.nodes()]

    # Calculate dynamic font sizes (as a dictionary for easy lookup by node)
    node_font_sizes = {node: MIN_FONT_SIZE + node_degrees[node] * FONT_DEGREE_SCALE_FACTOR for node in subgraph.nodes()}
    
    # Draw nodes
    node_colors = [subgraph.nodes[node]['color'] for node in subgraph.nodes()] 
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                            node_size=node_sizes, alpha=ALPHA_NODES) # Use dynamic node_sizes

    # Draw edges - scale width by weight
    edge_widths = [subgraph[u][v]['weight'] * EDGE_WIDTH_SCALE for u, v in subgraph.edges()] 
    nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=ALPHA_EDGES, edge_color=EDGE_COLOR)

    # Draw labels individually with dynamic font sizes
    for node, (x, y) in pos.items():
        # Shift labels upward as before
        label_x, label_y = x, y + 0.015 
        
        ax.text(label_x, label_y, node, 
                fontsize=node_font_sizes[node], 
                color="black", 
                ha='center', va='bottom') # Horizontal alignment to center, vertical to bottom of text
        
    plt.axis('off') # Hide axes

    # --- Legend removed as requested ---
    
    plt.tight_layout() # Adjust layout, no rect needed without legend
    
    # Output filename
    graph_filename = os.path.join(FIGURES_DIR, "fig_coincidence_graph.png")
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight') 
    print(f"\nCo-occurrence graph saved to: {graph_filename}")
    plt.show() # Display the plot
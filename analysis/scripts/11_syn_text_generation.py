#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os
import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import unicodedata

# --- Configuration ---
# Match the paths from your provided scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_MASTER_SHEET_PROCESSING_DIR = PROJECT_ROOT / "outputs" / "master_sheet_processing"
INPUTS_TFIDF_VECTORIZATION_DIR = PROJECT_ROOT / "outputs" / "TF_IDF_vectorization"

# --- Synthetic Generation Parameters ---
MORPHEME_CLASSES = ["Xihuitl", "Xochitl", "Quahuitl", "Patli", "Quilitl"]
# This ratio can be tuned. We will generate synthetic samples for under-represented classes.
SYNTHETIC_RATIO = 2.0
N_NEIGHBORS = 5
PCA_COMPONENTS = 50 # A reasonable number for dimensionality reduction

# --- Helper function to create multi-hot labels ---
def get_multi_hot_labels(df, morpheme_classes):
    """
    Generates a multi-hot encoded label matrix from a DataFrame's morphemes_raw_string.
    """
    labels = np.zeros((len(df), len(morpheme_classes)), dtype=np.float32)
    for i, morpheme_string in df['morphemes_raw_string'].fillna('').items():
        if isinstance(morpheme_string, str):
            found_morphemes = [entry.strip().split('/')[0] for entry in morpheme_string.split(';')
                               if entry.strip().split('/')[0] in morpheme_classes]
            for morpheme in found_morphemes:
                labels[i, morpheme_classes.index(morpheme)] = 1.0
    return labels

# --- Helper function for synthetic sample generation (SMOTE-like) ---
def generate_synthetic_samples(X, y, n_neighbors, synthetic_ratio, morpheme_classes):
    """
    Generates synthetic samples for under-represented classes.
    """
    synthetic_samples = []
    synthetic_labels = []

    # Iterate through each morpheme class
    for class_idx in range(len(morpheme_classes)):
        class_indices = np.where(y[:, class_idx] == 1)[0]
        num_class_samples = len(class_indices)

        if num_class_samples == 0:
            print(f"Warning: No samples found for class '{morpheme_classes[class_idx]}'. Skipping synthetic generation.")
            continue

        # Determine the number of synthetic samples to generate
        num_to_generate = int(num_class_samples * synthetic_ratio)
        if num_to_generate > 0:
            print(f"Generating {num_to_generate} synthetic samples for class '{morpheme_classes[class_idx]}'.")

            # Use NearestNeighbors on the PCA-transformed data
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(X[class_indices])

            for _ in tqdm(range(num_to_generate), desc=f"Class {morpheme_classes[class_idx]}"):
                # Randomly select a sample and a neighbor
                sample_idx_in_class = np.random.randint(0, num_class_samples)
                sample_pca = X[class_indices[sample_idx_in_class]]
                
                # Find its neighbors
                _, neighbors_indices = nn.kneighbors([sample_pca])
                neighbor_idx_in_class = np.random.choice(neighbors_indices[0][1:]) # Exclude the sample itself

                neighbor_pca = X[class_indices[neighbor_idx_in_class]]

                # Interpolate in the PCA space
                alpha = np.random.rand()
                synthetic_pca = sample_pca + alpha * (neighbor_pca - sample_pca)

                synthetic_samples.append(synthetic_pca)
                # Assign the original morpheme label to the new synthetic sample
                synthetic_labels.append(y[class_indices[sample_idx_in_class]])
    
    if not synthetic_samples:
        return None, None

    return np.array(synthetic_samples), np.array(synthetic_labels)

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Language-specific file paths ---
    file_paths = {
        'english': {
            'tfidf_matrix': INPUTS_TFIDF_VECTORIZATION_DIR / 'english_TF_IDF_vectorization_matrix.npz',
            'fasttext_embeddings': INPUTS_MASTER_SHEET_PROCESSING_DIR / "english_text_embeddings.npy",
            'text_column': 'aggregated_english_text',
        },
        'spanish': {
            'tfidf_matrix': INPUTS_TFIDF_VECTORIZATION_DIR / 'spanish_TF_IDF_vectorization_matrix.npz',
            'fasttext_embeddings': INPUTS_MASTER_SHEET_PROCESSING_DIR / "spanish_text_embeddings.npy",
            'text_column': 'aggregated_spanish_text',
        }
    }
    
    # Load raw data once to get the labels
    AGGREGATED_TEXTS_CSV = INPUTS_MASTER_SHEET_PROCESSING_DIR / "aggregated_plant_texts.csv"
    try:
        raw_df = pd.read_csv(AGGREGATED_TEXTS_CSV)
    except FileNotFoundError:
        print(f"Error: Required aggregated texts CSV not found at {AGGREGATED_TEXTS_CSV}.")
        sys.exit(1)

    for language in ['english', 'spanish']:
        print("\n" + "="*80)
        print(f"--- STARTING SYNTHETIC DATA GENERATION FOR {language.upper()} ---")
        print("="*80)
        
        # --- Output Directories ---
        SYNTHETIC_DATA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic_text_data" / language
        os.makedirs(SYNTHETIC_DATA_OUTPUT_DIR, exist_ok=True)
        print(f"Output directory for synthetic data: {SYNTHETIC_DATA_OUTPUT_DIR}")

        # Filter dataframe for the current language
        df_language = raw_df[raw_df[file_paths[language]['text_column']].fillna('').str.strip() != '']
        y_labels_language = get_multi_hot_labels(df_language, MORPHEME_CLASSES)

        # --- Process TF-IDF Vectors ---
        print("\n--- Processing TF-IDF Vectors ---")
        try:
            X_tfidf_real = load_npz(file_paths[language]['tfidf_matrix']).toarray()
            print(f"Loaded {language} TF-IDF matrix with shape: {X_tfidf_real.shape}")
        except FileNotFoundError:
            print(f"Error: TF-IDF matrix not found for {language} at {file_paths[language]['tfidf_matrix']}. Skipping TF-IDF generation for this language.")
            continue # Skip to the next language if a file is missing

        # Apply PCA to TF-IDF data
        print("Applying PCA to TF-IDF data...")
        pca_tfidf = PCA(n_components=PCA_COMPONENTS, random_state=42)
        X_tfidf_pca = pca_tfidf.fit_transform(X_tfidf_real)

        # Generate synthetic samples in the PCA space
        synthetic_samples_pca_tfidf, synthetic_labels_tfidf = generate_synthetic_samples(
            X_tfidf_pca, y_labels_language, N_NEIGHBORS, SYNTHETIC_RATIO, MORPHEME_CLASSES
        )

        if synthetic_samples_pca_tfidf is not None:
            # Inverse transform the synthetic samples back to the original feature space
            X_tfidf_synthetic = pca_tfidf.inverse_transform(synthetic_samples_pca_tfidf)
            
            # Combine real and synthetic data
            X_tfidf_combined = np.concatenate([X_tfidf_real, X_tfidf_synthetic], axis=0)
            y_tfidf_combined = np.concatenate([y_labels_language, synthetic_labels_tfidf], axis=0)
            is_real_flags_tfidf = np.array([True] * len(X_tfidf_real) + [False] * len(X_tfidf_synthetic))

            tfidf_final_data = {
                'X_vectors': X_tfidf_combined,
                'y_labels': y_tfidf_combined,
                'is_real_flags': is_real_flags_tfidf,
                'class_names': MORPHEME_CLASSES
            }
            with open(SYNTHETIC_DATA_OUTPUT_DIR / f"{language}_tfidf_synthetic_dataset.pkl", 'wb') as f:
                pickle.dump(tfidf_final_data, f)
            print(f"Saved combined TF-IDF dataset (real + synthetic) to {SYNTHETIC_DATA_OUTPUT_DIR / f'{language}_tfidf_synthetic_dataset.pkl'}")
        else:
            print(f"No synthetic TF-IDF samples were generated for {language}.")

        # --- Process FastText Embeddings ---
        print("\n--- Processing FastText Embeddings ---")
        try:
            X_fasttext_real = np.load(file_paths[language]['fasttext_embeddings'])
            # Reshape for PCA: flatten the sequence dimension
            num_samples, seq_len, embed_dim = X_fasttext_real.shape
            X_fasttext_real_flat = X_fasttext_real.reshape(num_samples, seq_len * embed_dim)
            print(f"Loaded {language} FastText embeddings with shape: {X_fasttext_real.shape}")
            print(f"Flattened for PCA to shape: {X_fasttext_real_flat.shape}")
        except FileNotFoundError:
            print(f"Error: FastText embeddings not found for {language} at {file_paths[language]['fasttext_embeddings']}. Skipping FastText generation for this language.")
            continue # Skip to the next language if a file is missing

        # Apply PCA to FastText data
        print("Applying PCA to flattened FastText data...")
        pca_fasttext = PCA(n_components=PCA_COMPONENTS, random_state=42)
        X_fasttext_pca = pca_fasttext.fit_transform(X_fasttext_real_flat)

        # Generate synthetic samples in the PCA space
        synthetic_samples_pca_fasttext, synthetic_labels_fasttext = generate_synthetic_samples(
            X_fasttext_pca, y_labels_language, N_NEIGHBORS, SYNTHETIC_RATIO, MORPHEME_CLASSES
        )

        if synthetic_samples_pca_fasttext is not None:
            # Inverse transform the synthetic samples back to the original feature space
            X_fasttext_synthetic_flat = pca_fasttext.inverse_transform(synthetic_samples_pca_fasttext)
            X_fasttext_synthetic = X_fasttext_synthetic_flat.reshape(-1, seq_len, embed_dim)
            
            # Combine real and synthetic data
            X_fasttext_combined = np.concatenate([X_fasttext_real, X_fasttext_synthetic], axis=0)
            y_fasttext_combined = np.concatenate([y_labels_language, synthetic_labels_fasttext], axis=0)
            is_real_flags_fasttext = np.array([True] * len(X_fasttext_real) + [False] * len(X_fasttext_synthetic))

            fasttext_final_data = {
                'X_vectors': X_fasttext_combined,
                'y_labels': y_fasttext_combined,
                'is_real_flags': is_real_flags_fasttext,
                'class_names': MORPHEME_CLASSES,
                'max_sequence_length': seq_len,
                'embedding_dim': embed_dim
            }
            with open(SYNTHETIC_DATA_OUTPUT_DIR / f"{language}_fasttext_synthetic_dataset.pkl", 'wb') as f:
                pickle.dump(fasttext_final_data, f)
            print(f"Saved combined FastText dataset (real + synthetic) to {SYNTHETIC_DATA_OUTPUT_DIR / f'{language}_fasttext_synthetic_dataset.pkl'}")
        else:
            print(f"No synthetic FastText samples were generated for {language}.")

    print("\n--- Synthetic Text Data Generation Completed ---")
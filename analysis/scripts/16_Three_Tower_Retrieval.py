import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pickle
import cv2
import os
from pathlib import Path
import json
import random
from collections import defaultdict

# --- Configuration and File Paths ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT_DIR / "outputs"
DATA_DIR = PROJECT_ROOT_DIR / "data"

# Image and Metadata Paths
SYNTHETIC_DATA_OUTPUT_DIR = OUTPUTS_DIR / "synthetic_leaf_data"
SYNTHETIC_METADATA_PATH = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"
NAHUATL_NAMES_PATH = DATA_DIR / "FOR NAHUATL REVIEW - Nahuatl names.csv"

# Text Data Paths
ENGLISH_FASTTEXT_PKL = OUTPUTS_DIR / "synthetic_text_data" / "english" / "english_fasttext_synthetic_dataset.pkl"
AGGREGATED_TEXTS_CSV = OUTPUTS_DIR / "master_sheet_processing" / "aggregated_plant_texts.csv"

# NEW: Path to the generated node embeddings
GRAPH_EMBEDDINGS_PKL = OUTPUTS_DIR / "three_tower_embeddings" / "nahuatl_node_embeddings.pkl"

# Model and training parameters
IMG_WIDTH, IMG_HEIGHT = 256, 256
EMBEDDING_DIM = 300
EMBEDDING_SIZE = 128
GRAPH_EMBEDDING_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
N_SPLITS = 5
SEED = 42

# Output directory for this experiment
EXPERIMENT_NAME = "three_tower_synthetic_kfold_retrieval"
EXPERIMENT_OUTPUT_DIR = OUTPUTS_DIR / EXPERIMENT_NAME
os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_OUTPUT_DIR / "models", exist_ok=True)
os.makedirs(EXPERIMENT_OUTPUT_DIR / "metrics", exist_ok=True)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Set seed for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

set_seed(SEED)

# --- Step 1: Loading and Aligning Data ---
print("--- Step 1: Loading and Aligning Data ---")

# Load image metadata
synthetic_df = pd.read_csv(SYNTHETIC_METADATA_PATH)
real_image_metadata = synthetic_df[synthetic_df['is_real'] == True]
real_image_metadata = real_image_metadata[real_image_metadata['is_processed_valid'] == True]

# Load text data and graph embeddings
with open(ENGLISH_FASTTEXT_PKL, 'rb') as f:
    text_data_dict = pickle.load(f)
raw_texts_df = pd.read_csv(AGGREGATED_TEXTS_CSV)
with open(GRAPH_EMBEDDINGS_PKL, 'rb') as f:
    graph_embeddings_dict = pickle.load(f)

# Align real image, text, and graph data by 'official_name'
aligned_real_data = pd.merge(real_image_metadata, raw_texts_df,
                             left_on='class_label', right_on='official_name',
                             how='inner', suffixes=('_img', '_txt'))

# Filter out classes that don't have a graph embedding
aligned_real_data['official_name_clean'] = aligned_real_data['official_name'].str.strip().str.lower()
aligned_real_data = aligned_real_data[aligned_real_data['official_name_clean'].isin(graph_embeddings_dict.keys())].reset_index(drop=True)

# Filter out classes with fewer samples than n_splits for valid stratification
counts = aligned_real_data['official_name'].value_counts()
valid_classes = counts[counts >= N_SPLITS].index
aligned_real_data = aligned_real_data[aligned_real_data['official_name'].isin(valid_classes)].reset_index(drop=True)

print(f"Number of aligned REAL samples for K-Fold after filtering: {len(aligned_real_data)}")

# --- Re-populate data arrays from the filtered dataframe ---
X_real_images = []
X_real_text = []
X_real_graph = []
y_real_labels_plant_ids = []
plant_id_to_label = {plant_id: i for i, plant_id in enumerate(aligned_real_data['official_name'].unique())}

for idx, row in aligned_real_data.iterrows():
    ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_ect']
    mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_mask']
    
    ect_img = cv2.imread(str(ect_path), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if ect_img is not None and mask_img is not None:
        ect_img = cv2.resize(ect_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask_img = cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        stacked_img = np.stack([ect_img, mask_img], axis=0)
        X_real_images.append(stacked_img)

        text_index = raw_texts_df[raw_texts_df['official_name'] == row['official_name']].index[0]
        X_real_text.append(text_data_dict['X_vectors'][text_index])

        graph_embedding = graph_embeddings_dict[row['official_name_clean']]
        X_real_graph.append(graph_embedding)

        y_real_labels_plant_ids.append(plant_id_to_label[row['official_name']])

X_real_images = np.array(X_real_images, dtype=np.float32)
X_real_text = np.array(X_real_text, dtype=np.float32)
X_real_graph = np.array(X_real_graph, dtype=np.float32)
y_real_labels_plant_ids = np.array(y_real_labels_plant_ids, dtype=np.int64)

synthetic_image_metadata = synthetic_df[synthetic_df['is_real'] == False]
synthetic_text_indices = np.where(text_data_dict['is_real_flags'] == False)[0]

NUM_SYNTHETIC_RATIO = 10
num_synthetic_to_load = len(X_real_images) * NUM_SYNTHETIC_RATIO
num_available_synthetic_images = len(synthetic_image_metadata)
num_available_synthetic_texts = len(synthetic_text_indices)
num_to_sample = min(num_synthetic_to_load, num_available_synthetic_images, num_available_synthetic_texts)

np.random.seed(SEED)
synthetic_image_indices = np.random.choice(num_available_synthetic_images, size=num_to_sample, replace=False)
synthetic_text_indices_for_images = np.random.choice(synthetic_text_indices, size=num_to_sample, replace=False)

X_synthetic_images = []
for i in synthetic_image_indices:
    row = synthetic_image_metadata.iloc[i]
    ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_ect']
    mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_mask']
    
    ect_img = cv2.imread(str(ect_path), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if ect_img is not None and mask_img is not None:
        ect_img = cv2.resize(ect_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask_img = cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        stacked_img = np.stack([ect_img, mask_img], axis=0)
        X_synthetic_images.append(stacked_img)
X_synthetic_images = np.array(X_synthetic_images, dtype=np.float32)

X_synthetic_text = text_data_dict['X_vectors'][synthetic_text_indices_for_images]

print(f"Loaded {len(X_synthetic_images)} synthetic image and text pairs for training.")

# --- Step 2: Custom PyTorch Dataset and DataLoader (Modified for 3 modalities) ---
class ThreeTowerRetrievalDataset(Dataset):
    def __init__(self, images, texts, graphs, labels):
        self.images = torch.from_numpy(images)
        self.texts = torch.from_numpy(texts)
        self.graphs = torch.from_numpy(graphs)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.graphs[idx], self.labels[idx]

# --- Step 3: Building the Three-Tower Retrieval Model with PyTorch ---
class ThreeTowerRetrievalModel(nn.Module):
    def __init__(self, embedding_size):
        super(ThreeTowerRetrievalModel, self).__init__()
        self.image_tower = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.image_output_size = self._get_image_output_size()
        self.image_projection = nn.Linear(self.image_output_size, embedding_size)

        self.text_tower = nn.Sequential(
            nn.Conv1d(EMBEDDING_DIM, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten()
        )
        self.text_projection = nn.Linear(128, embedding_size)
        
        self.graph_projection = nn.Linear(GRAPH_EMBEDDING_SIZE, embedding_size)
        
    def _get_image_output_size(self):
        dummy_input = torch.randn(1, 2, IMG_WIDTH, IMG_HEIGHT)
        output = self.image_tower(dummy_input)
        return output.size(1)

    def forward_image(self, image_input):
        features = self.image_tower(image_input)
        embeddings = self.image_projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward_text(self, text_input):
        features = self.text_tower(text_input.permute(0, 2, 1))
        embeddings = self.text_projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward_graph(self, graph_input):
        embeddings = self.graph_projection(graph_input)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, image_input, text_input, graph_input):
        img_emb = self.forward_image(image_input)
        txt_emb = self.forward_text(text_input)
        graph_emb = self.forward_graph(graph_input)
        return img_emb, txt_emb, graph_emb

# --- Step 4: Multi-Modal Contrastive Loss Function ---
class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(MultiModalContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img_emb, txt_emb, graph_emb):
        loss_img_txt = self.contrastive_loss_pair(img_emb, txt_emb)
        loss_img_graph = self.contrastive_loss_pair(img_emb, graph_emb)
        loss_txt_graph = self.contrastive_loss_pair(txt_emb, graph_emb)
        
        return loss_img_txt + loss_img_graph + loss_txt_graph

    def contrastive_loss_pair(self, emb1, emb2):
        positive_similarity = F.cosine_similarity(emb1, emb2)
        emb1_expanded = emb1.unsqueeze(1)
        emb2_expanded = emb2.unsqueeze(0)
        negative_similarity = F.cosine_similarity(emb1_expanded, emb2_expanded, dim=2)
        negative_similarity = negative_similarity.fill_diagonal_(0.0)
        loss_positive = (1 - positive_similarity).pow(2).mean()
        loss_negative = torch.clamp(negative_similarity - self.margin, min=0).pow(2).mean()
        return loss_positive + loss_negative

# --- Step 5: K-Fold Cross-Validation Loop (Modified for 3 modalities) ---
print("\n--- Step 5: Starting K-Fold Cross-Validation with Synthetic Data ---")
if len(X_real_images) == 0:
    print("Error: No real images were loaded after filtering. Cannot perform K-Fold validation.")
elif len(np.unique(y_real_labels_plant_ids)) < N_SPLITS:
    print(f"Error: The number of unique classes ({len(np.unique(y_real_labels_plant_ids))}) is less than the number of splits ({N_SPLITS}). Cannot perform stratified K-fold. Please adjust N_SPLITS.")
else:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_indices, test_indices) in enumerate(skf.split(X_real_images, y_real_labels_plant_ids)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        set_seed(SEED + fold)

        X_train_real_img = X_real_images[train_indices]
        X_train_real_txt = X_real_text[train_indices]
        X_train_real_graph = X_real_graph[train_indices]

        X_test_real_img = X_real_images[test_indices]
        X_test_real_txt = X_real_text[test_indices]
        X_test_real_graph = X_real_graph[test_indices]
        
        # Combine real and synthetic for training
        X_train_img_combined = np.concatenate([X_train_real_img, X_synthetic_images], axis=0)
        X_train_txt_combined = np.concatenate([X_train_real_txt, X_synthetic_text], axis=0)
        X_train_graph_combined = np.concatenate([X_train_real_graph, X_train_real_graph], axis=0)
        
        # New: Pass labels to the dataset for saving embeddings
        train_dataset = ThreeTowerRetrievalDataset(X_train_img_combined, X_train_txt_combined, X_train_graph_combined, np.concatenate([y_real_labels_plant_ids[train_indices], np.full(len(X_synthetic_images), -1)]))
        test_dataset = ThreeTowerRetrievalDataset(X_test_real_img, X_test_real_txt, X_test_real_graph, y_real_labels_plant_ids[test_indices])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = ThreeTowerRetrievalModel(EMBEDDING_SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = MultiModalContrastiveLoss(margin=0.5)

        print(f"Training model for fold {fold+1} on {len(X_train_img_combined)} samples (Real+Synthetic)...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for images, texts, graphs, _ in train_loader:
                images, texts, graphs = images.to(device), texts.to(device), graphs.to(device)
                optimizer.zero_grad()
                img_emb, txt_emb, graph_emb = model(images, texts, graphs)
                loss = criterion(img_emb, txt_emb, graph_emb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        # --- MODIFIED 'evaluate_fold_metrics' function to save embeddings ---
        def calculate_metrics(similarity_matrix, direction_name):
            recall_at_k = {}
            ap_sum = 0
            num_queries = similarity_matrix.size(0)
            
            for k in [1, 5, 10]:
                correct_retrievals = 0
                for i in range(num_queries):
                    if 'query' in direction_name:
                        topk_indices = torch.topk(similarity_matrix[i], k).indices
                    else: # 'target' direction
                        topk_indices = torch.topk(similarity_matrix[:, i], k).indices
                    if i in topk_indices:
                        correct_retrievals += 1
                score = correct_retrievals / num_queries
                chance_score = k / num_queries
                recall_at_k[f'Recall@{k}'] = {
                    'score': score, 
                    'above_chance_x': score / chance_score if chance_score > 0 else 0
                }

            ap_sum = 0
            for i in range(num_queries):
                if 'query' in direction_name:
                    retrieved_indices = torch.argsort(similarity_matrix[i], descending=True)
                else:
                    retrieved_indices = torch.argsort(similarity_matrix[:, i], descending=True)

                retrieved_is_relevant = torch.zeros(num_queries, dtype=torch.float32)
                retrieved_is_relevant[retrieved_indices == i] = 1.0
                num_relevant = 1
                
                precision_at_k = torch.cumsum(retrieved_is_relevant, dim=0) / torch.arange(1, num_queries + 1)
                ap = torch.sum(precision_at_k * retrieved_is_relevant) / num_relevant
                ap_sum += ap
            
            mAP = ap_sum / num_queries
            
            return {'Recall@K': recall_at_k, 'mAP': mAP.item()}

        def evaluate_fold_metrics(model, test_loader, device, fold_num, output_dir):
            model.eval()
            all_img_embeddings = []
            all_txt_embeddings = []
            all_graph_embeddings = []
            all_labels = []

            with torch.no_grad():
                for images, texts, graphs, labels in test_loader:
                    images, texts, graphs = images.to(device), texts.to(device), graphs.to(device)
                    img_emb, txt_emb, graph_emb = model(images, texts, graphs)
                    
                    all_img_embeddings.append(img_emb.cpu())
                    all_txt_embeddings.append(txt_emb.cpu())
                    all_graph_embeddings.append(graph_emb.cpu())
                    all_labels.extend(labels.tolist())

            all_img_embeddings = torch.cat(all_img_embeddings)
            all_txt_embeddings = torch.cat(all_txt_embeddings)
            all_graph_embeddings = torch.cat(all_graph_embeddings)
            
            # Save the embeddings and labels for later visualization
            embeddings_data = {
                'image_embeddings': all_img_embeddings.numpy(),
                'text_embeddings': all_txt_embeddings.numpy(),
                'graph_embeddings': all_graph_embeddings.numpy(),
                'labels': np.array(all_labels)
            }
            
            with open(os.path.join(output_dir, f"fold_{fold_num}_embeddings.pkl"), 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            # Calculate similarity matrices for each pair
            sim_img_txt = F.cosine_similarity(all_img_embeddings.unsqueeze(1), all_txt_embeddings.unsqueeze(0), dim=2)
            sim_img_graph = F.cosine_similarity(all_img_embeddings.unsqueeze(1), all_graph_embeddings.unsqueeze(0), dim=2)
            sim_txt_graph = F.cosine_similarity(all_txt_embeddings.unsqueeze(1), all_graph_embeddings.unsqueeze(0), dim=2)

            # Calculate and store pairwise metrics
            metrics = {}
            metrics['image_to_text'] = calculate_metrics(sim_img_txt, 'query_image_target_text')
            metrics['text_to_image'] = calculate_metrics(sim_img_txt.T, 'query_text_target_image')
            metrics['image_to_graph'] = calculate_metrics(sim_img_graph, 'query_image_target_graph')
            metrics['graph_to_image'] = calculate_metrics(sim_img_graph.T, 'query_graph_target_image')
            metrics['text_to_graph'] = calculate_metrics(sim_txt_graph, 'query_text_target_graph')
            metrics['graph_to_text'] = calculate_metrics(sim_txt_graph.T, 'query_graph_target_text')
            
            return metrics
        
        # MODIFIED CALL: Pass fold_num and output_dir to the evaluation function
        fold_metrics.append(evaluate_fold_metrics(model, test_loader, device, fold, EXPERIMENT_OUTPUT_DIR))

    print("\n--- Consolidating Results Across All Folds ---")
    all_metrics = {
        'image_to_text': defaultdict(list),
        'text_to_image': defaultdict(list),
        'image_to_graph': defaultdict(list),
        'graph_to_image': defaultdict(list),
        'text_to_graph': defaultdict(list),
        'graph_to_text': defaultdict(list),
    }

    for fold in fold_metrics:
        for metric_key, metric_data in fold.items():
            all_metrics[metric_key]['mAP'].append(metric_data['mAP'])
            for k in [1, 5, 10]:
                all_metrics[metric_key][f'Recall@{k}_score'].append(metric_data['Recall@K'][f'Recall@{k}']['score'])
                all_metrics[metric_key][f'Recall@{k}_above_chance'].append(metric_data['Recall@K'][f'Recall@{k}']['above_chance_x'])
    
    # Calculate final averages and std deviations
    final_avg_metrics = defaultdict(dict)
    num_test_samples = len(X_real_images) // N_SPLITS
    random_chance_at_1 = 1 / num_test_samples
    
    for metric_key, data in all_metrics.items():
        final_avg_metrics[metric_key]['Random_Chance_at_1'] = random_chance_at_1
        final_avg_metrics[metric_key]['mAP_avg'] = np.mean(data['mAP'])
        final_avg_metrics[metric_key]['mAP_std'] = np.std(data['mAP'])
        for k in [1, 5, 10]:
            final_avg_metrics[metric_key][f'Recall@{k}_score_avg'] = np.mean(data[f'Recall@{k}_score'])
            final_avg_metrics[metric_key][f'Recall@{k}_score_std'] = np.std(data[f'Recall@{k}_score'])
            final_avg_metrics[metric_key][f'Recall@{k}_above_chance_avg'] = np.mean(data[f'Recall@{k}_above_chance'])

    print("\n--- Final K-Fold Retrieval Metrics with Three Towers ---")
    for metric_key, data in final_avg_metrics.items():
        print(f"\n--- {metric_key.replace('_', ' ').title()} Retrieval ---")
        print(f"Random Chance @1: {data['Random_Chance_at_1']:.4f}")
        for k in [1, 5, 10]:
            avg_score = data[f'Recall@{k}_score_avg']
            std_score = data[f'Recall@{k}_score_std']
            above_chance = data[f'Recall@{k}_above_chance_avg']
            print(f" Recall@{k}: {avg_score:.4f} +/- {std_score:.4f} ({above_chance:.2f}x above chance)")
        print(f"Mean Average Precision (mAP): {data['mAP_avg']:.4f} +/- {data['mAP_std']:.4f}")

    metrics_path = EXPERIMENT_OUTPUT_DIR / "metrics" / "kfold_retrieval_three_tower_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_avg_metrics, f, indent=4)
    print(f"\nFinal K-fold retrieval metrics with three towers saved to {metrics_path}")

print("\nScript completed successfully.")
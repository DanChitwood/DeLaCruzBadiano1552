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

# Text Data Paths (using English as the primary text tower)
ENGLISH_FASTTEXT_PKL = OUTPUTS_DIR / "synthetic_text_data" / "english" / "english_fasttext_synthetic_dataset.pkl"
AGGREGATED_TEXTS_CSV = OUTPUTS_DIR / "master_sheet_processing" / "aggregated_plant_texts.csv"

# Model and training parameters
IMG_WIDTH, IMG_HEIGHT = 256, 256
EMBEDDING_DIM = 300
EMBEDDING_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
N_SPLITS = 5
SEED = 42

# Output directory for this experiment
EXPERIMENT_NAME = "two_tower_synthetic_kfold_retrieval"
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
nahuatl_df = pd.read_csv(NAHUATL_NAMES_PATH)
nahuatl_df = nahuatl_df[['official_name', 'label']]
real_image_metadata = synthetic_df[synthetic_df['is_real'] == True]
real_image_metadata = real_image_metadata[real_image_metadata['is_processed_valid'] == True]
print(f"Loaded {len(real_image_metadata)} real image samples.")

# Load text data, including synthetic text and original plant names
with open(ENGLISH_FASTTEXT_PKL, 'rb') as f:
    text_data_dict = pickle.load(f)

raw_texts_df = pd.read_csv(AGGREGATED_TEXTS_CSV)
raw_texts_df['is_real_text'] = raw_texts_df['official_name'].isin(real_image_metadata['class_label'])

# Align real image and text data by 'official_name'
aligned_real_data = pd.merge(real_image_metadata, raw_texts_df,
                             left_on='class_label', right_on='official_name',
                             how='inner', suffixes=('_img', '_txt'))

# Filter out classes with fewer samples than n_splits for valid stratification
counts = aligned_real_data['official_name'].value_counts()
valid_classes = counts[counts >= N_SPLITS].index
aligned_real_data = aligned_real_data[aligned_real_data['official_name'].isin(valid_classes)].reset_index(drop=True)

print(f"Number of aligned REAL samples for K-Fold after filtering: {len(aligned_real_data)}")

# --- Re-populate data arrays from the filtered dataframe ---
X_real_images = []
X_real_text = []
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
        y_real_labels_plant_ids.append(plant_id_to_label[row['official_name']])

X_real_images = np.array(X_real_images, dtype=np.float32)
X_real_text = np.array(X_real_text, dtype=np.float32)
y_real_labels_plant_ids = np.array(y_real_labels_plant_ids, dtype=np.int64)

synthetic_image_metadata = synthetic_df[synthetic_df['is_real'] == False]
synthetic_text_indices = np.where(text_data_dict['is_real_flags'] == False)[0]

NUM_SYNTHETIC_RATIO = 10
num_synthetic_to_load = len(X_real_images) * NUM_SYNTHETIC_RATIO
num_available_synthetic_images = len(synthetic_image_metadata)
num_available_synthetic_texts = len(synthetic_text_indices)
num_to_sample = min(num_synthetic_to_load, num_available_synthetic_images, num_available_synthetic_texts)

print(f"Number of synthetic samples to be used for training: {num_to_sample}")
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

# --- Step 2: Custom PyTorch Dataset and DataLoader ---
class TwoTowerRetrievalDataset(Dataset):
    def __init__(self, images, texts):
        self.images = torch.from_numpy(images)
        self.texts = torch.from_numpy(texts)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]

# --- Step 3: Building the Two-Tower Retrieval Model with PyTorch ---
class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, embedding_size):
        super(TwoTowerRetrievalModel, self).__init__()
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

    def forward(self, image_input, text_input):
        img_emb, txt_emb = self.forward_image(image_input), self.forward_text(text_input)
        return img_emb, txt_emb

# --- Step 4: Contrastive Loss Function ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img_emb, txt_emb):
        positive_similarity = F.cosine_similarity(img_emb, txt_emb)
        img_emb_expanded = img_emb.unsqueeze(1)
        txt_emb_expanded = txt_emb.unsqueeze(0)
        negative_similarity = F.cosine_similarity(img_emb_expanded, txt_emb_expanded, dim=2)
        negative_similarity = negative_similarity.fill_diagonal_(0.0)
        loss_positive = (1 - positive_similarity).pow(2).mean()
        loss_negative = torch.clamp(negative_similarity - self.margin, min=0).pow(2).mean()
        return loss_positive + loss_negative

# --- Step 5: K-Fold Cross-Validation Loop ---
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
        X_test_real_img = X_real_images[test_indices]
        X_test_real_txt = X_real_text[test_indices]
        
        X_train_img_combined = np.concatenate([X_train_real_img, X_synthetic_images], axis=0)
        X_train_txt_combined = np.concatenate([X_train_real_txt, X_synthetic_text], axis=0)

        train_dataset = TwoTowerRetrievalDataset(X_train_img_combined, X_train_txt_combined)
        test_dataset = TwoTowerRetrievalDataset(X_test_real_img, X_test_real_txt)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = TwoTowerRetrievalModel(EMBEDDING_SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = ContrastiveLoss(margin=0.5)

        print(f"Training model for fold {fold+1} on {len(X_train_img_combined)} samples (Real+Synthetic)...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for images, texts in train_loader:
                images, texts = images.to(device), texts.to(device)
                optimizer.zero_grad()
                img_emb, txt_emb = model(images, texts)
                loss = criterion(img_emb, txt_emb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        def calculate_metrics(similarity_matrix, direction_name):
            recall_at_k = {}
            ap_sum = 0
            num_queries = similarity_matrix.size(0)
            
            for k in [1, 5, 10]:
                correct_retrievals = 0
                for i in range(num_queries):
                    if direction_name == 'image_to_text':
                        topk_indices = torch.topk(similarity_matrix[i], k).indices
                    else:
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
                if direction_name == 'image_to_text':
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

        def evaluate_fold_metrics(model, test_loader, device):
            model.eval()
            all_img_embeddings = []
            all_txt_embeddings = []

            with torch.no_grad():
                for images, texts in test_loader:
                    images, texts = images.to(device), texts.to(device)
                    img_emb = model.forward_image(images)
                    txt_emb = model.forward_text(texts)
                    all_img_embeddings.append(img_emb.cpu())
                    all_txt_embeddings.append(txt_emb.cpu())

            all_img_embeddings = torch.cat(all_img_embeddings)
            all_txt_embeddings = torch.cat(all_txt_embeddings)

            similarity_matrix = F.cosine_similarity(all_img_embeddings.unsqueeze(1), all_txt_embeddings.unsqueeze(0), dim=2)
            
            image_to_text_metrics = calculate_metrics(similarity_matrix, 'image_to_text')
            text_to_image_metrics = calculate_metrics(similarity_matrix.T, 'text_to_image')
            
            return image_to_text_metrics, text_to_image_metrics
        
        img_to_text_metrics, txt_to_img_metrics = evaluate_fold_metrics(model, test_loader, device)
        fold_metrics.append({'image_to_text': img_to_text_metrics, 'text_to_image': txt_to_img_metrics})

    print("\n--- Consolidating Results Across All Folds ---")
    final_metrics = {
        'image_to_text': { 'Recall@K': defaultdict(list), 'mAP': [] },
        'text_to_image': { 'Recall@K': defaultdict(list), 'mAP': [] }
    }

    for fold in fold_metrics:
        final_metrics['image_to_text']['mAP'].append(fold['image_to_text']['mAP'])
        for k in [1, 5, 10]:
            final_metrics['image_to_text']['Recall@K'][str(k)].append({
                'score': fold['image_to_text']['Recall@K'][f'Recall@{k}']['score'],
                'above_chance_x': fold['image_to_text']['Recall@K'][f'Recall@{k}']['above_chance_x']
            })
            
        final_metrics['text_to_image']['mAP'].append(fold['text_to_image']['mAP'])
        for k in [1, 5, 10]:
            final_metrics['text_to_image']['Recall@K'][str(k)].append({
                'score': fold['text_to_image']['Recall@K'][f'Recall@{k}']['score'],
                'above_chance_x': fold['text_to_image']['Recall@K'][f'Recall@{k}']['above_chance_x']
            })
    
    # Restructuring for averaging
    avg_metrics_temp = {
        'image_to_text': { 'Recall@K': defaultdict(list), 'mAP': [], 'above_chance_x': defaultdict(list)},
        'text_to_image': { 'Recall@K': defaultdict(list), 'mAP': [], 'above_chance_x': defaultdict(list)}
    }
    
    for fold in final_metrics['image_to_text']['mAP']:
        avg_metrics_temp['image_to_text']['mAP'].append(fold)
    for k in [1, 5, 10]:
        for fold_results in final_metrics['image_to_text']['Recall@K'][str(k)]:
            avg_metrics_temp['image_to_text']['Recall@K'][str(k)].append(fold_results['score'])
            avg_metrics_temp['image_to_text']['above_chance_x'][str(k)].append(fold_results['above_chance_x'])
    
    for fold in final_metrics['text_to_image']['mAP']:
        avg_metrics_temp['text_to_image']['mAP'].append(fold)
    for k in [1, 5, 10]:
        for fold_results in final_metrics['text_to_image']['Recall@K'][str(k)]:
            avg_metrics_temp['text_to_image']['Recall@K'][str(k)].append(fold_results['score'])
            avg_metrics_temp['text_to_image']['above_chance_x'][str(k)].append(fold_results['above_chance_x'])

    avg_metrics = {
        'image_to_text': {'mAP_avg': np.mean(avg_metrics_temp['image_to_text']['mAP']), 
                          'mAP_std': np.std(avg_metrics_temp['image_to_text']['mAP']),
                          'Recall@K_metrics': defaultdict(dict)},
        'text_to_image': {'mAP_avg': np.mean(avg_metrics_temp['text_to_image']['mAP']), 
                          'mAP_std': np.std(avg_metrics_temp['text_to_image']['mAP']),
                          'Recall@K_metrics': defaultdict(dict)}
    }

    num_test_samples = len(X_real_images) // N_SPLITS
    avg_metrics['image_to_text']['Random_Chance_at_1'] = 1 / num_test_samples
    avg_metrics['text_to_image']['Random_Chance_at_1'] = 1 / num_test_samples

    for k in [1, 5, 10]:
        k_str = str(k)
        avg_metrics['image_to_text']['Recall@K_metrics'][k_str]['score_avg'] = np.mean(avg_metrics_temp['image_to_text']['Recall@K'][k_str])
        avg_metrics['image_to_text']['Recall@K_metrics'][k_str]['score_std'] = np.std(avg_metrics_temp['image_to_text']['Recall@K'][k_str])
        avg_metrics['image_to_text']['Recall@K_metrics'][k_str]['above_chance_x_avg'] = np.mean(avg_metrics_temp['image_to_text']['above_chance_x'][k_str])
        
        avg_metrics['text_to_image']['Recall@K_metrics'][k_str]['score_avg'] = np.mean(avg_metrics_temp['text_to_image']['Recall@K'][k_str])
        avg_metrics['text_to_image']['Recall@K_metrics'][k_str]['score_std'] = np.std(avg_metrics_temp['text_to_image']['Recall@K'][k_str])
        avg_metrics['text_to_image']['Recall@K_metrics'][k_str]['above_chance_x_avg'] = np.mean(avg_metrics_temp['text_to_image']['above_chance_x'][k_str])

    print("\n--- Final K-Fold Retrieval Metrics with Synthetic Training Data ---")
    print("Image to Text Retrieval:")
    print(f"Random Chance @1: {avg_metrics['image_to_text']['Random_Chance_at_1']:.4f}")
    for k in [1, 5, 10]:
        k_str = str(k)
        avg_score = avg_metrics['image_to_text']['Recall@K_metrics'][k_str]['score_avg']
        std_score = avg_metrics['image_to_text']['Recall@K_metrics'][k_str]['score_std']
        above_chance_x_avg = avg_metrics['image_to_text']['Recall@K_metrics'][k_str]['above_chance_x_avg']
        print(f" Recall@{k}: {avg_score:.4f} +/- {std_score:.4f} ({above_chance_x_avg:.2f}x above chance)")
    print(f"Mean Average Precision (mAP): {avg_metrics['image_to_text']['mAP_avg']:.4f} +/- {avg_metrics['image_to_text']['mAP_std']:.4f}")

    print("\nText to Image Retrieval:")
    print(f"Random Chance @1: {avg_metrics['text_to_image']['Random_Chance_at_1']:.4f}")
    for k in [1, 5, 10]:
        k_str = str(k)
        avg_score = avg_metrics['text_to_image']['Recall@K_metrics'][k_str]['score_avg']
        std_score = avg_metrics['text_to_image']['Recall@K_metrics'][k_str]['score_std']
        above_chance_x_avg = avg_metrics['text_to_image']['Recall@K_metrics'][k_str]['above_chance_x_avg']
        print(f" Recall@{k}: {avg_score:.4f} +/- {std_score:.4f} ({above_chance_x_avg:.2f}x above chance)")
    print(f"Mean Average Precision (mAP): {avg_metrics['text_to_image']['mAP_avg']:.4f} +/- {avg_metrics['text_to_image']['mAP_std']:.4f}")

    metrics_path = EXPERIMENT_OUTPUT_DIR / "metrics" / "kfold_retrieval_synthetic_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"\nFinal K-fold retrieval metrics with synthetic data saved to {metrics_path}")

print("\nScript completed successfully.")
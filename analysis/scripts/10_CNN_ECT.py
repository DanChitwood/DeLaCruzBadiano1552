#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
# Using KFold for multi-label, as StratifiedKFold does not support it out-of-the-box
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, multilabel_confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle
from pathlib import Path
import sys
import json

# PyTorch Imports
import torch
import torch.nn as tnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import cv2
from PIL import Image

############################
### CONFIGURATION (ALL PARAMETERS UP FRONT) ###
############################

# --- Project Structure Configuration ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_OUTPUT_BASE_DIR = PROJECT_ROOT_DIR / "outputs" / "cnn_analysis"

# --- General Model Training Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
K_FOLDS = 5
PATIENCE = 10 # Early stopping patience (for validation loss)
LR_SCHEDULER_PATIENCE = 5 # Patience for ReduceLROnPlateau
LR_SCHEDULER_FACTOR = 0.1 # Factor by which to reduce LR
MODEL_IDENTIFIER = 'Illustrated_Plants_Morpheme_CNN' # Unique identifier for this model/run

# --- Data Input Configuration ---
FINAL_PREPARED_DATA_FILE = PROJECT_ROOT_DIR / "outputs" / "synthetic_leaf_data" / "final_cnn_dataset.pkl"
NAHUATL_MASTER_DATA_FILE = PROJECT_ROOT_DIR / "data" / "FOR NAHUATL REVIEW - Nahuatl names.csv"

# --- Morpheme Classification Configuration ---
# The 5 morpheme classes we are now targeting
MORPHEME_CLASSES = ["Xihuitl", "Xochitl", "Quahuitl", "Patli", "Quilitl"]
# The target column for classification, e.g., 'Leaf_Class'
target_column_used_for_data = 'Nahuatl_Morphemes'

# --- Output Directories Setup ---
MODEL_SAVE_DIR = EXPERIMENT_OUTPUT_BASE_DIR / "trained_models"
METRICS_SAVE_DIR = MODEL_SAVE_DIR / "metrics_output"
CONFUSION_MATRIX_DATA_DIR = MODEL_SAVE_DIR / "confusion_matrix_data"
GRAD_CAM_OUTPUT_DIR = MODEL_SAVE_DIR / "grad_cam_images"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DATA_DIR, exist_ok=True)
os.makedirs(GRAD_CAM_OUTPUT_DIR, exist_ok=True)
print(f"Base Project directory set to: {PROJECT_ROOT_DIR}")
print(f"Experiment outputs will be saved to: {EXPERIMENT_OUTPUT_BASE_DIR}")
print(f"Model outputs (e.g., .pth files) will be saved to: {MODEL_SAVE_DIR}")
print(f"Metrics, Confusion Matrix data, and Grad-CAM images will be saved within: {MODEL_SAVE_DIR}")

# Grad-CAM specific configurations
NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT = 5 # Number of real samples per class to average for Grad-CAM

# --- Global Results Storage ---
results_storage = {}

##########################
### DEVICE SETUP ###
##########################

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device for training: {device}")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Random seeds set for reproducibility.")

###########################
### DATA LOADING & LABEL PREPARATION ###
###########################

print("\n--- Loading data from FINAL_PREPARED_DATA_FILE ---")
try:
    with open(FINAL_PREPARED_DATA_FILE, 'rb') as f:
        final_data = pickle.load(f)

    X_images = final_data['X_images'] # (N, H, W, 2) numpy array
    y_labels_encoded_123 = final_data['y_labels_encoded'] # (N,) numpy array, old 123-class labels
    is_real_flags = final_data['is_real_flags'] # (N,) boolean numpy array
    class_names_123 = final_data['class_names'] # List of 123 string class names
    image_size_tuple = final_data['image_size'] # (H, W) tuple, e.g., (256, 256)
    num_channels = final_data['num_channels'] # int, should be 2

    # --- NEW: Load master CSV to get morpheme labels ---
    nahuatl_names_df = pd.read_csv(NAHUATL_MASTER_DATA_FILE)

    # We need a way to link the image data (from final_cnn_dataset.pkl) to the morpheme data
    # The 'class_names' from the pickle file are the 'official_name's in the master CSV.
    # The old `y_labels_encoded_123` array maps to these class_names.
    # We will use this to build our new morpheme labels.

    # 1. Map old class names to their morpheme labels (multi-label)
    morpheme_mapping = {}
    for name in class_names_123:
        # Find the corresponding row in the master CSV
        row = nahuatl_names_df[nahuatl_names_df['official_name'] == name]
        if not row.empty:
            label_string = row['label'].iloc[0] if pd.notna(row['label'].iloc[0]) else ''
            # Split the string by ';' and extract the morpheme part (e.g., 'Xochitl')
            morphemes_for_name = [
                entry.strip().split('/')[0] for entry in label_string.split(';')
                if entry.strip().split('/')[0] in MORPHEME_CLASSES
            ]
            morpheme_mapping[name] = morphemes_for_name
        else:
            morpheme_mapping[name] = [] # No morpheme labels found for this class name

    # 2. Create the new y_labels_encoded (multi-hot encoded) array
    num_morpheme_classes = len(MORPHEME_CLASSES)
    y_labels_encoded_morphemes = np.zeros((len(X_images), num_morpheme_classes), dtype=np.float32)

    for i in range(len(X_images)):
        old_class_encoded = y_labels_encoded_123[i]
        old_class_name = class_names_123[old_class_encoded]
        morphemes = morpheme_mapping[old_class_name]
        for morpheme in morphemes:
            morpheme_idx = MORPHEME_CLASSES.index(morpheme)
            y_labels_encoded_morphemes[i, morpheme_idx] = 1.0

    print(f"Loaded image data shape: {X_images.shape}")
    print(f"Number of classes: {num_morpheme_classes} ({', '.join(MORPHEME_CLASSES)})")
    print(f"Image size: {image_size_tuple}")
    print(f"Number of channels: {num_channels}")
    print(f"Number of real samples: {np.sum(is_real_flags)}")
    print(f"Number of synthetic samples: {np.sum(~is_real_flags)}")
    print(f"Data will be processed for classification of: '{target_column_used_for_data}'")

except FileNotFoundError:
    print(f"Error: Data file not found. Ensure {FINAL_PREPARED_DATA_FILE} and {NAHUATL_MASTER_DATA_FILE} exist.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)

# --- PyTorch Data Preparation ---
# Permute dimensions from (N, H, W, C) to (N, C, H, W) for PyTorch CNN input
X_images_tensor = torch.from_numpy(X_images).float().permute(0, 3, 1, 2)
# y_labels_encoded is now multi-hot, so it needs to be a float tensor
y_morpheme_tensor = torch.from_numpy(y_labels_encoded_morphemes).float()
is_real_flag_tensor = torch.from_numpy(is_real_flags).bool()

print(f"Tensor image data shape (after permute): {X_images_tensor.shape}")
print(f"Tensor label data shape (multi-hot): {y_morpheme_tensor.shape}")


# ---------------------------------------------------------------------------- #
#                                                                              #
#                        PYTORCH DATASET & MODEL                               #
#                                                                              #
# ---------------------------------------------------------------------------- #

class LeafDataset(Dataset):
    """
    A custom PyTorch Dataset for leaf images.
    Returns image tensor, MULTI-LABEL class vector, and a boolean flag indicating if it's a real sample.
    """
    def __init__(self, images, labels, is_real_flags):
        self.images = images
        self.labels = labels
        self.is_real_flags = is_real_flags

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.is_real_flags[idx]

class LeafCNN(tnn.Module):
    def __init__(self, num_classes, image_size, num_input_channels):
        super(LeafCNN, self).__init__()
        self.features = tnn.Sequential(
            tnn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1),
            tnn.BatchNorm2d(32),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),
            tnn.Conv2d(32, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),
            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            temp_features_model = self.features.to(device)
            dummy_input = torch.zeros(1, num_input_channels, image_size[0], image_size[1]).to(device)
            flattened_size = temp_features_model(dummy_input).view(1, -1).shape[1]
            temp_features_model.to("cpu")
        self.classifier = tnn.Sequential(
            tnn.Flatten(),
            tnn.Linear(flattened_size, 512),
            tnn.ReLU(),
            tnn.Dropout(0.5),
            # The final layer output size must match the number of morpheme classes
            tnn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def save_model_checkpoint(model, optimizer, epoch, accuracy, model_identifier, target_column, fold_idx):
    filepath = MODEL_SAVE_DIR / f"{model_identifier}_fold{fold_idx}_best_{target_column}.pth"
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(state, filepath)
    print(f"  --> Saved best model for Fold {fold_idx} (Accuracy: {accuracy:.4f}) at {filepath}")
    return filepath

# ---------------------------------------------------------------------------- #
#                                                                              #
#        PYTORCH CNN TRAINING AND EVALUATION (Ensemble with K-Fold)            #
#                                                                              #
# ---------------------------------------------------------------------------- #

print(f"\n--- Performing PyTorch CNN with {K_FOLDS}-Fold Cross-Validation on Morphemes ---")

real_original_indices_global = torch.where(is_real_flag_tensor)[0].cpu().numpy()
X_original_images_for_skf = X_images_tensor[real_original_indices_global]
y_original_for_skf = y_morpheme_tensor[real_original_indices_global]

# KFold is used instead of StratifiedKFold because the labels are multi-label.
# We will still split the REAL samples to evaluate generalizability.
kf_pytorch = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

all_predictions_logits = []
all_predictions_binary = [] # Store binary predictions for multi-label metrics
saved_model_paths_per_fold = [None] * K_FOLDS

# --- Calculate class weights for imbalanced dataset ---
# We now compute weights for each of the 5 morpheme classes.
# Note: This is an approximation for multi-label. More sophisticated methods exist.
all_training_labels_for_weights = y_morpheme_tensor.cpu().numpy()
class_counts = np.sum(all_training_labels_for_weights, axis=0)
total_samples = len(all_training_labels_for_weights)
# Formula for balanced weights in multi-label: N / (C * N_c) where N is total samples, C is total classes, N_c is count of class c
morpheme_class_weights = total_samples / (num_morpheme_classes * class_counts)
class_weights_tensor = torch.tensor(morpheme_class_weights, dtype=torch.float).to(device)
print(f"\nCalculated morpheme class weights: {class_weights_tensor.cpu().numpy()}")

for fold_idx, (train_original_real_indices, val_original_real_indices) in enumerate(kf_pytorch.split(X_original_images_for_skf.cpu().numpy())):
    print(f"\n--- Fold {fold_idx + 1}/{K_FOLDS} ---")

    X_val_img_fold = X_original_images_for_skf[val_original_real_indices]
    y_val_fold = y_original_for_skf[val_original_real_indices]
    val_dataset = LeafDataset(X_val_img_fold, y_val_fold, torch.ones_like(y_val_fold[:,0], dtype=torch.bool)) # Note the change in label indexing
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    synthetic_indices = torch.where(~is_real_flag_tensor)[0].cpu().numpy()
    global_real_train_indices = real_original_indices_global[train_original_real_indices]
    all_training_indices_global = np.concatenate((global_real_train_indices, synthetic_indices))

    X_train_img_fold_tensor = X_images_tensor[all_training_indices_global]
    y_train_fold_tensor = y_morpheme_tensor[all_training_indices_global]
    is_real_train_fold_tensor = is_real_flag_tensor[all_training_indices_global]

    train_dataset = LeafDataset(X_train_img_fold_tensor, y_train_fold_tensor, is_real_train_fold_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialize model, loss function, optimizer, and learning rate scheduler
    model = LeafCNN(num_classes=num_morpheme_classes, image_size=image_size_tuple, num_input_channels=num_channels).to(device)
    # The loss function for multi-label classification is BCEWithLogitsLoss
    criterion = tnn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_overall_accuracy_for_saving_this_fold = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_true = []
            val_pred = []
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # For multi-label, we apply a threshold (e.g., 0.5) to the sigmoid output
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                val_true.append(labels.cpu().numpy())
                val_pred.append(predictions.cpu().numpy())
        
            avg_train_loss = running_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            val_true_arr = np.concatenate(val_true, axis=0)
            val_pred_arr = np.concatenate(val_pred, axis=0)

            # Calculate a multi-label accuracy score (e.g., subset accuracy or F1)
            # F1 score is a good robust metric for multi-label
            val_accuracy = f1_score(val_true_arr, val_pred_arr, average='micro', zero_division=0)
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_accuracy:.4f} (Real Samples)")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Save model checkpoint
            if val_accuracy > best_overall_accuracy_for_saving_this_fold:
                best_overall_accuracy_for_saving_this_fold = val_accuracy
                path_to_saved_model = save_model_checkpoint(model, optimizer, epoch, best_overall_accuracy_for_saving_this_fold, MODEL_IDENTIFIER, target_column_used_for_data, fold_idx)
                saved_model_paths_per_fold[fold_idx] = path_to_saved_model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    model.load_state_dict(best_model_wts)
    print(f"Fold {fold_idx + 1} training complete. Best validation loss for fold: {best_val_loss:.4f}")

    # Predict logits and binary predictions for ALL real samples using the best model
    model.eval()
    fold_predictions_logits = []
    fold_predictions_binary = []

    real_dataset_for_pred = LeafDataset(X_original_images_for_skf, y_original_for_skf, torch.ones_like(y_original_for_skf[:,0], dtype=torch.bool))
    real_loader_for_pred = DataLoader(real_dataset_for_pred, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images_batch, _, _ in real_loader_for_pred:
            images_batch = images_batch.to(device)
            outputs = model(images_batch)
            fold_predictions_logits.append(outputs.cpu().numpy())
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            fold_predictions_binary.append(predictions.cpu().numpy())

    all_predictions_logits.append(np.concatenate(fold_predictions_logits, axis=0))
    all_predictions_binary.append(np.concatenate(fold_predictions_binary, axis=0))

# ---------------------------------------------------------------------------- #
#                                                                              #
#              FINAL ENSEMBLE EVALUATION ON REAL SAMPLES ONLY                  #
#                                                                              #
# ---------------------------------------------------------------------------- #

print("\n--- Final Ensemble Evaluation on ALL REAL Samples ---")

# Average the logits from all K folds for the final ensemble prediction
averaged_logits = np.mean(np.array(all_predictions_logits), axis=0)
final_predictions_binary = (torch.sigmoid(torch.from_numpy(averaged_logits)) > 0.5).long().numpy()

final_true_labels_binary = y_original_for_skf.cpu().numpy()

# Calculate and print overall F1 score (a good multi-label metric)
micro_f1 = f1_score(final_true_labels_binary, final_predictions_binary, average='micro', zero_division=0)
macro_f1 = f1_score(final_true_labels_binary, final_predictions_binary, average='macro', zero_division=0)
print(f"\n--- Overall Micro F1 Score ({MODEL_IDENTIFIER} Ensemble, Real Samples ONLY): {micro_f1:.4f} ---")
print(f"--- Overall Macro F1 Score ({MODEL_IDENTIFIER} Ensemble, Real Samples ONLY): {macro_f1:.4f} ---")

# Generate and print the classification report
print(f"\n--- Classification Report ({MODEL_IDENTIFIER} Ensemble, Real Samples ONLY) ---")
report_dict = classification_report(final_true_labels_binary, final_predictions_binary, target_names=MORPHEME_CLASSES, zero_division=0, output_dict=True)
print(classification_report(final_true_labels_binary, final_predictions_binary, target_names=MORPHEME_CLASSES, zero_division=0))

# --- Save Classification Report to JSON ---
metrics_output_path = METRICS_SAVE_DIR / f"{MODEL_IDENTIFIER}_classification_report_{target_column_used_for_data}.json"
with open(metrics_output_path, 'w') as f:
    json.dump(report_dict, f, indent=4)
print(f"Classification report saved to: {metrics_output_path}")

# Compute and plot confusion matrix (one-vs-all for multi-label)
cm_real_pt_multilabel = multilabel_confusion_matrix(final_true_labels_binary, final_predictions_binary)

# --- Plot and save the Confusion Matrices (one for each morpheme class) ---
plt.style.use('default') # Reset style for multi-plot grid
fig, axes = plt.subplots(nrows=1, ncols=len(MORPHEME_CLASSES), figsize=(4 * len(MORPHEME_CLASSES), 4))
if len(MORPHEME_CLASSES) == 1:
    axes = [axes] # Ensure axes is iterable even for a single class

for i, cm in enumerate(cm_real_pt_multilabel):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                xticklabels=['Not Predicted', 'Predicted'],
                yticklabels=['Not True', 'True'])
    axes[i].set_title(f'CM for {MORPHEME_CLASSES[i]}', fontsize=12)
    axes[i].set_xlabel('Predicted label')
    axes[i].set_ylabel('True label')

plt.tight_layout()
plt.savefig(METRICS_SAVE_DIR / f"{MODEL_IDENTIFIER}_ConfusionMatrices_{target_column_used_for_data}.png", dpi=300)
plt.show()

# --- Store results in a global dictionary ---
if target_column_used_for_data not in results_storage:
    results_storage[target_column_used_for_data] = {
        'class_counts': {},
        'model_metrics': {}
    }
print(f"Global target_column_used_for_data for this session: '{target_column_used_for_data}'")

MODEL_NAME = MODEL_IDENTIFIER
class_counts_per_morpheme = np.sum(final_true_labels_binary, axis=0)
for i, morpheme_name in enumerate(MORPHEME_CLASSES):
    results_storage[target_column_used_for_data]['class_counts'][morpheme_name] = int(class_counts_per_morpheme[i])
print(f"Class counts populated for '{target_column_used_for_data}'.")

results_storage[target_column_used_for_data]['model_metrics'][MODEL_NAME] = {
    'precision': {cls: report_dict[cls]['precision'] for cls in MORPHEME_CLASSES},
    'recall': {cls: report_dict[cls]['recall'] for cls in MORPHEME_CLASSES},
    'f1-score': {cls: report_dict[cls]['f1-score'] for cls in MORPHEME_CLASSES},
    'micro avg f1': micro_f1,
    'macro avg f1': macro_f1,
    'weighted avg precision': report_dict['weighted avg']['precision'],
    'weighted avg recall': report_dict['weighted avg']['recall'],
    'weighted avg f1-score': report_dict['weighted avg']['f1-score'],
}
print(f"Metrics for '{MODEL_NAME}' stored in results_storage for '{target_column_used_for_data}'.")
print("\n--- Current contents of results_storage (should include new model metrics) ---")
print(results_storage)


# ---------------------------------------------------------------------------- #
#                                                                              #
#                         GRAD-CAM VISUALIZATION                               #
#                                                                              #
# ---------------------------------------------------------------------------- #

print(f"\n--- Generating Average Grad-CAM Visualizations for {MODEL_IDENTIFIER} (Model from Fold 0) ---")

if len(saved_model_paths_per_fold) > 0 and saved_model_paths_per_fold[0] is not None and os.path.exists(saved_model_paths_per_fold[0]):
    model_to_visualize_path = saved_model_paths_per_fold[0]
    cam_model = LeafCNN(num_classes=len(MORPHEME_CLASSES), image_size=image_size_tuple, num_input_channels=num_channels).to(device)
    checkpoint = torch.load(model_to_visualize_path, map_location=device)
    cam_model.load_state_dict(checkpoint['model_state_dict'])
    cam_model.eval()

    target_layer = cam_model.features[-3]
    
    # Redefined Grad-CAM classes and functions to be included in the main script flow
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            found_layer = False
            for name, module in self.model.named_modules():
                if module is self.target_layer:
                    module.register_forward_hook(self._save_activation)
                    module.register_backward_hook(self._save_gradient)
                    found_layer = True
                    break
            if not found_layer:
                raise ValueError(f"Target layer {target_layer} not found in model named modules.")

        def _save_activation(self, module, input, output):
            self.activations = output

        def _save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def __call__(self, input_tensor, target_class=None):
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            if target_class is None:
                # For multi-label, we can't use argmax. We'll use the class with the highest logit
                target_class = output.argmax(dim=1).item()
            
            one_hot = torch.zeros_like(output).to(device)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            gradients = self.gradients[0].cpu().data.numpy()
            activations = self.activations[0].cpu().data.numpy()
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
            return cam
            
    def show_cam_on_black_background(cam_heatmap, original_image_tensor, image_size_tuple):
        ect_channel = original_image_tensor[1, :, :].cpu().numpy()
        ect_channel_display = ect_channel - ect_channel.min()
        if ect_channel_display.max() > 0:
            ect_channel_display = ect_channel_display / ect_channel_display.max()
        else:
            ect_channel_display = np.zeros_like(ect_channel_display)
        img_display_base = np.stack([ect_channel_display, ect_channel_display, ect_channel_display], axis=-1)
        img_display_base = np.uint8(255 * img_display_base)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255
        alpha = 0.5
        final_cam_img = np.uint8(255 * (heatmap_colored * alpha + np.float32(img_display_base) / 255 * (1-alpha)))
        return final_cam_img

    grad_cam = GradCAM(cam_model, target_layer)
    average_class_heatmaps = {}
    real_indices_by_class = {cls_idx: [] for cls_idx in range(len(MORPHEME_CLASSES))}
    
    # We now populate the indices based on the NEW morpheme labels.
    for i in range(len(y_original_for_skf)):
        morpheme_vector = y_original_for_skf[i]
        for morpheme_idx in range(len(morpheme_vector)):
            if morpheme_vector[morpheme_idx] == 1:
                real_indices_by_class[morpheme_idx].append(i)

    print("Calculating average Grad-CAM heatmaps per class...")
    for class_idx in range(len(MORPHEME_CLASSES)):
        class_name = MORPHEME_CLASSES[class_idx]
        class_samples_indices = real_indices_by_class[class_idx]

        if not class_samples_indices:
            print(f"  No real samples for morpheme class '{class_name}'. Skipping average Grad-CAM.")
            average_class_heatmaps[class_idx] = None
            continue

        summed_heatmap = np.zeros(image_size_tuple, dtype=np.float32)
        count_for_average = 0
        samples_for_cam = np.random.choice(class_samples_indices, min(NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT, len(class_samples_indices)), replace=False)

        for sample_idx in samples_for_cam:
            image_tensor = X_original_images_for_skf[sample_idx]
            input_image_for_cam = image_tensor.unsqueeze(0).to(device)
            heatmap = grad_cam(input_image_for_cam, target_class=class_idx)
            summed_heatmap += heatmap
            count_for_average += 1
            
        if count_for_average > 0:
            avg_heatmap = summed_heatmap / count_for_average
            avg_heatmap = avg_heatmap - np.min(avg_heatmap)
            if np.max(avg_heatmap) == 0:
                avg_heatmap = np.zeros_like(avg_heatmap)
            else:
                avg_heatmap = avg_heatmap / np.max(avg_heatmap)
            average_class_heatmaps[class_idx] = avg_heatmap
            print(f"  Calculated average for morpheme class: '{class_name}' ({count_for_average} samples)")
        else:
            average_class_heatmaps[class_idx] = None

    num_plots_total = len(MORPHEME_CLASSES)
    num_cols_grid = math.ceil(math.sqrt(num_plots_total))
    num_rows_grid = math.ceil(num_plots_total / num_cols_grid)
    fig_width = num_cols_grid * 3.0
    fig_height = num_rows_grid * 3.5
    sns.set_style("white")
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    print(f"\nPlotting average Grad-CAMs in a {num_rows_grid}x{num_cols_grid} grid...")
    for i in range(len(MORPHEME_CLASSES)):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(MORPHEME_CLASSES[i], fontsize=10)
        avg_heatmap = average_class_heatmaps[i]
        if avg_heatmap is not None:
            if real_indices_by_class[i]:
                example_image_tensor = X_original_images_for_skf[real_indices_by_class[i][0]]
                cam_image_on_background = show_cam_on_black_background(avg_heatmap, example_image_tensor, image_size_tuple)
                ax.imshow(cam_image_on_background)
                individual_cam_output_path = GRAD_CAM_OUTPUT_DIR / f"{MODEL_IDENTIFIER}_GradCAM_{MORPHEME_CLASSES[i]}.png"
                fig_single = plt.figure(figsize=(image_size_tuple[0]/100, image_size_tuple[1]/100), dpi=100)
                ax_single = fig_single.add_subplot(111)
                ax_single.imshow(cam_image_on_background)
                ax_single.set_axis_off()
                ax_single.set_position([0,0,1,1])
                fig_single.savefig(individual_cam_output_path, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig_single)
                print(f"  Saved individual Grad-CAM for class '{MORPHEME_CLASSES[i]}' to: {individual_cam_output_path}")
            else:
                ax.text(0.5, 0.5, 'No Samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=10)

    for j in range(num_plots_total, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(f'Average Grad-CAM Visualizations ({MODEL_IDENTIFIER}, Target: {target_column_used_for_data})', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(GRAD_CAM_OUTPUT_DIR / f"{MODEL_IDENTIFIER}_AverageGradCAM_{target_column_used_for_data}.png", dpi=300)
    plt.show()

else:
    print("Skipping Grad-CAM visualization because the model for Fold 0 was not found or saved.")
    print(f"Expected model path: {saved_model_paths_per_fold[0] if len(saved_model_paths_per_fold) > 0 else 'N/A'}")

print("\n--- CNN Training and Evaluation Script Completed ---")
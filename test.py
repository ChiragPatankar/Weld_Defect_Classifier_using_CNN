import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import json
import os
from os import walk

# Define directory paths - match train.py
BASE_DIR = "./data"
TRAIN_DIR = "./data/train-20250310T201516Z-001/train"
TEST_DIR = os.path.join(BASE_DIR, "test-20250310T201517Z-001/test")
TRAIN_JSON = "data/train-20250310T201516Z-001/train/train.json"
TEST_JSON = os.path.join(TEST_DIR, "test.json")
MODEL_PATH = os.path.join(BASE_DIR, "modelv5.pt")

# Verify directories exist
print(f"Checking if directories exist:")
print(f"TEST_DIR: {os.path.exists(TEST_DIR)}")
print(f"TEST_JSON: {os.path.exists(TEST_JSON)}")
print(f"MODEL_PATH: {os.path.exists(MODEL_PATH)}")

# Set PyTorch configurations
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(False)  # No gradients needed for testing

# Parameters
mean = 0.2069
std = 0.1471


# Define Network Architecture (identical to train.py)
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(102400 // 200, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.1),
            nn.Linear(128, 6),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 102400 // 200)
        x = self.classifier(x)
        return x


def get_num_correct(pred, label):
    return pred.argmax(dim=1).eq(label).sum().item()


# Check for CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare test dataset
transform_test = transforms.Compose([
    transforms.CenterCrop((800, 800)),
    transforms.ToTensor(),
])

print("\nLoading test dataset...")
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform_test)
original_test_labels_dict = test_dataset.class_to_idx
print(f"Found {len(test_dataset)} test images in {len(original_test_labels_dict)} classes")

# Load test JSON labels
print("Loading test JSON labels...")
with open(TEST_JSON, 'r') as f:
    label_test = json.load(f)

# Debug: Print JSON structure
print("\nJSON structure sample:")
for i, (key, value) in enumerate(list(label_test.items())[:3]):
    print(f"  {key}: {value}")

# Debug: Print folder structure
print("\nFolder structure sample:")
for i, (folder, idx) in enumerate(list(original_test_labels_dict.items())[:3]):
    print(f"  {folder}: {idx}")

# Create test label mapping using the same approach as in train.py
test_label_map_dict = {}
for folder, idx in original_test_labels_dict.items():
    # Try different key formats to find a match
    folder_key = folder

    # Method 1: Direct match
    if folder_key in label_test:
        test_label_map_dict[idx] = label_test[folder_key]
        print(f"Direct match: {folder_key} -> {label_test[folder_key]}")
        continue

    # Method 2: Check if folder name is in any JSON key
    found = False
    for json_key in label_test.keys():
        # Extract the folder name from JSON keys
        json_folder = json_key.split('/')[-1] if '/' in json_key else json_key
        if folder_key == json_folder:
            test_label_map_dict[idx] = label_test[json_key]
            found = True
            print(f"Path match: {folder_key} -> {json_key} -> {label_test[json_key]}")
            break

    # Method 3: Try fuzzy matching
    if not found:
        for json_key in label_test.keys():
            if folder_key in json_key or folder_key.replace(" ", "") in json_key.replace("/", ""):
                test_label_map_dict[idx] = label_test[json_key]
                found = True
                print(f"Fuzzy match: {folder_key} -> {json_key} -> {label_test[json_key]}")
                break

    if not found:
        print(f"WARNING: No matching test label found for folder '{folder_key}'")
        test_label_map_dict[idx] = -1

print("\nTest label mapping:")
print(test_label_map_dict)

# Handle missing test mappings
if -1 in test_label_map_dict.values():
    missing_folders = [folder for folder, idx in original_test_labels_dict.items()
                       if test_label_map_dict[idx] == -1]
    print(f"ERROR: Missing test labels for folders: {missing_folders}")

    # Print possible matches to help debugging
    print("\nPossible JSON keys that might match:")
    for missing_folder in missing_folders:
        similar_keys = [k for k in label_test.keys() if any(part in k for part in missing_folder.split())]
        print(f"  For '{missing_folder}', possible matches: {similar_keys}")

    # Manual mapping for test folders - add your mappings here if needed
    manual_test_mapping = {
        # Example: 'folder_name': corresponding_label_value
    }

    for folder, idx in original_test_labels_dict.items():
        if test_label_map_dict[idx] == -1 and folder in manual_test_mapping:
            test_label_map_dict[idx] = manual_test_mapping[folder]
            print(f"Manual test mapping: {folder} -> {manual_test_mapping[folder]}")

    # Check if we still have missing mappings
    if -1 in test_label_map_dict.values():
        remaining_missing = [folder for folder, idx in original_test_labels_dict.items()
                             if test_label_map_dict[idx] == -1]
        print(f"\nERROR: Still missing test labels for folders: {remaining_missing}")
        print("Please update the manual test mapping in the code.")
        raise ValueError("Missing test label mappings")

# Create 2-class mapping for binary classification
print("\nCreating binary classification mapping...")
test_label_map_dict2 = {}
for idx, label in test_label_map_dict.items():
    # 0 = Good Weld, 1-5 = Defective Weld
    test_label_map_dict2[idx] = 0 if label == 0 else 1

print("Binary classification mapping:")
print(test_label_map_dict2)

# Load model
print("\nLoading model from", MODEL_PATH)
net = Network()
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()  # Set to evaluation mode


# Define function for binary classification
def get_num_correct2(pred, true):
    prediction = pred.argmax(dim=1)
    # Convert to binary: 0 stays 0, everything else becomes 1
    binary_pred = [0 if val.item() == 0 else 1 for val in prediction]
    binary_pred = torch.tensor(binary_pred, device=true.device)
    return binary_pred.eq(true).sum().item()


# Test the model with 6-class classification
print("\n--- 6-Class Classification Evaluation ---")
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
total_correct = 0
total = 0

for i, batch in enumerate(test_loader):
    images, label_original = batch

    # Map original labels to true labels
    labels = torch.tensor([test_label_map_dict[lab.item()] for lab in label_original])

    # Forward pass
    predictions = net(images.to(device))

    # Calculate accuracy
    batch_correct = get_num_correct(predictions, labels.to(device))
    total_correct += batch_correct
    total += len(labels)

    # Print batch progress
    batch_accuracy = 100 * batch_correct / len(labels)
    print(f'Test Batch {i + 1}/{len(test_loader)}, Batch Accuracy: {batch_accuracy:.2f}%')

# Final test accuracy
test_accuracy = 100 * total_correct / len(test_dataset)
print(f'\n6-Class Test Evaluation Complete. Final Test Accuracy: {test_accuracy:.2f}%')

# Test the model with 2-class classification (good vs defective)
print("\n--- 2-Class Classification Evaluation ---")
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
total_correct = 0
total = 0

for i, batch in enumerate(test_loader):
    images, label_original = batch

    # Map original labels to binary labels (0 = good, 1 = defective)
    labels = torch.tensor([test_label_map_dict2[lab.item()] for lab in label_original])

    # Forward pass
    predictions = net(images.to(device))

    # Calculate binary accuracy
    batch_correct = get_num_correct2(predictions, labels.to(device))
    total_correct += batch_correct
    total += len(labels)

    # Print batch progress
    batch_accuracy = 100 * batch_correct / len(labels)
    print(f'Test Batch {i + 1}/{len(test_loader)}, Binary Accuracy: {batch_accuracy:.2f}%')

# Final binary test accuracy
binary_test_accuracy = 100 * total_correct / len(test_dataset)
print(f'\n2-Class Test Evaluation Complete. Final Binary Test Accuracy: {binary_test_accuracy:.2f}%')

# Sample visualization
print("\n--- Sample Visualization ---")
# Create a smaller batch for visualization
viz_batch_size = 10
viz_loader = DataLoader(test_dataset, batch_size=viz_batch_size, shuffle=True)

# Define label dictionaries for visualization
defect_labels = {
    0: 'Good Weld',
    1: 'Burn through',
    2: 'Contamination',
    3: 'Lack of fusion',
    4: 'Misalignment',
    5: 'Lack of penetration'
}

binary_labels = {
    0: 'Good Weld',
    1: 'Defective Weld'
}

# Get a batch of images
images, label_original = next(iter(viz_loader))
pred = net(images.to(device))
pred_classes = pred.argmax(dim=1)

# Prepare true labels
true_labels = torch.tensor([test_label_map_dict[lab.item()] for lab in label_original])
true_binary = torch.tensor([test_label_map_dict2[lab.item()] for lab in label_original])

# Prepare binary predictions
pred_binary = torch.tensor([0 if val.item() == 0 else 1 for val in pred_classes])

# Display results
print("\n--- Sample Predictions ---")
for i in range(len(images)):
    print(f'Image {i + 1}:')
    print(f'  True Label: {defect_labels[true_labels[i].item()]} (Class {true_labels[i].item()})')
    print(f'  Predicted: {defect_labels[pred_classes[i].item()]} (Class {pred_classes[i].item()})')
    print(f'  Binary True: {binary_labels[true_binary[i].item()]}')
    print(f'  Binary Predicted: {binary_labels[pred_binary[i].item()]}')
    print(f'  Correct: {"Yes" if pred_classes[i] == true_labels[i] else "No"}')
    print(f'  Binary Correct: {"Yes" if pred_binary[i] == true_binary[i] else "No"}')
    print()

# Option to save predictions to a CSV file
print("\nSaving predictions to results.csv...")
all_predictions = []
all_true_labels = []
all_image_paths = []

# Get all predictions
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
for images, label_original in test_loader:
    # Get predictions
    pred = net(images.to(device))
    pred_classes = pred.argmax(dim=1).cpu().numpy()

    # Get true labels
    true_labels = [test_label_map_dict[lab.item()] for lab in label_original]

    # Get image paths
    batch_paths = [test_dataset.samples[idx][0] for idx in
                   range(len(all_true_labels), len(all_true_labels) + len(true_labels))]

    # Store data
    all_predictions.extend(pred_classes)
    all_true_labels.extend(true_labels)
    all_image_paths.extend(batch_paths)

# Create DataFrame
results_df = pd.DataFrame({
    'Image': all_image_paths,
    'True_Label': all_true_labels,
    'Predicted_Label': all_predictions,
    'Correct': [p == t for p, t in zip(all_predictions, all_true_labels)],
    'True_Defect': [defect_labels[t] for t in all_true_labels],
    'Predicted_Defect': [defect_labels[p] for p in all_predictions]
})

# Save to CSV
results_df.to_csv('results.csv', index=False)
print("Predictions saved to results.csv")

print("\nTest script completed successfully!")
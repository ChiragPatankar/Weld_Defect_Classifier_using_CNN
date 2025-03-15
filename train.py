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

# Define directory paths
BASE_DIR = "./data"
TRAIN_DIR = "./data/train-20250310T201516Z-001/train"
TEST_DIR = os.path.join(BASE_DIR, "test-20250310T201517Z-001/test")
TRAIN_JSON = "data/train-20250310T201516Z-001/train/train.json"
TEST_JSON = os.path.join(TEST_DIR, "test.json")

# Verify directories exist
print(f"Checking if directories exist:")
print(f"TRAIN_DIR: {os.path.exists(TRAIN_DIR)}")
print(f"TEST_DIR: {os.path.exists(TEST_DIR)}")
print(f"TRAIN_JSON: {os.path.exists(TRAIN_JSON)}")
print(f"TEST_JSON: {os.path.exists(TEST_JSON)}")

# Set PyTorch configurations
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

# Parameters
mean = 0.2069
std = 0.1471

# Define transformations
transform = transforms.Compose([
    transforms.CenterCrop((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the dataset
print("Loading training dataset...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
original_labels_dict = train_dataset.class_to_idx
print(f"Found {len(train_dataset)} images in {len(original_labels_dict)} classes")

# Load the JSON label file
print("Loading JSON labels...")
with open(TRAIN_JSON, 'r') as f:
    label_train = json.load(f)

# Debug: Print JSON structure
print("\nJSON structure sample:")
for i, (key, value) in enumerate(list(label_train.items())[:3]):
    print(f"  {key}: {value}")

# Debug: Print folder structure
print("\nFolder structure sample:")
for i, (folder, idx) in enumerate(list(original_labels_dict.items())[:3]):
    print(f"  {folder}: {idx}")

# Create a more flexible mapping between folder names and labels
label_map_dict = {}
for folder, idx in original_labels_dict.items():
    # Try different key formats to find a match
    folder_key = folder

    # Method 1: Direct match
    if folder_key in label_train:
        label_map_dict[idx] = label_train[folder_key]
        print(f"Direct match: {folder_key} -> {label_train[folder_key]}")
        continue

    # Method 2: Check if folder name is in any JSON key
    found = False
    for json_key in label_train.keys():
        # Extract the folder name from JSON keys
        json_folder = json_key.split('/')[-1] if '/' in json_key else json_key
        if folder_key == json_folder:
            label_map_dict[idx] = label_train[json_key]
            found = True
            print(f"Path match: {folder_key} -> {json_key} -> {label_train[json_key]}")
            break

    # Method 3: Try fuzzy matching
    if not found:
        for json_key in label_train.keys():
            if folder_key in json_key or folder_key.replace(" ", "") in json_key.replace("/", ""):
                label_map_dict[idx] = label_train[json_key]
                found = True
                print(f"Fuzzy match: {folder_key} -> {json_key} -> {label_train[json_key]}")
                break

    if not found:
        print(f"WARNING: No matching label found for folder '{folder_key}'")
        # You could set a default label or raise an error
        label_map_dict[idx] = -1  # Using -1 as a marker for "no label found"

print("\nFinal label mapping:")
print(label_map_dict)

# Exit if any mapping is missing
if -1 in label_map_dict.values():
    missing_folders = [folder for folder, idx in original_labels_dict.items()
                       if label_map_dict[idx] == -1]
    print(f"ERROR: Missing labels for folders: {missing_folders}")

    # Print possible matches to help debugging
    print("\nPossible JSON keys that might match:")
    for missing_folder in missing_folders:
        similar_keys = [k for k in label_train.keys() if any(part in k for part in missing_folder.split())]
        print(f"  For '{missing_folder}', possible matches: {similar_keys}")

    # Instead of raising an error, let's try a manual mapping
    print("\nAttempting manual mapping for missing folders...")

    # This is a placeholder for manual mapping based on your specific dataset
    # You'll need to adjust this based on your actual folder names and JSON keys
    manual_mapping = {
        # Example: 'folder_name': corresponding_label_value
        # '170815-133921-Al 2mm': 0,  # Adjust this based on your data
    }

    for folder, idx in original_labels_dict.items():
        if label_map_dict[idx] == -1 and folder in manual_mapping:
            label_map_dict[idx] = manual_mapping[folder]
            print(f"Manual mapping: {folder} -> {manual_mapping[folder]}")

    # Check if we still have missing mappings
    if -1 in label_map_dict.values():
        remaining_missing = [folder for folder, idx in original_labels_dict.items()
                             if label_map_dict[idx] == -1]
        print(f"\nERROR: Still missing labels for folders: {remaining_missing}")
        print("Please update the manual mapping in the code.")
        raise ValueError("Missing label mappings")


# Network Architecture
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

# Create and move model to device
net = Network()
net.to(device)

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-3)

print(f"Total images: {len(train_dataset)}")
print(f"Total batches: {len(train_loader)}")


# Training loop
def train_model(epochs=10, target_accuracy=90):
    epoch = 0
    best_accuracy = 0

    while epoch < epochs and best_accuracy < target_accuracy:
        total_loss = 0
        total_correct = 0

        for i, batch in enumerate(train_loader):
            images, label_original = batch

            # Map original labels to true labels
            labels = torch.tensor([label_map_dict[lab.item()] for lab in label_original])

            # Forward pass
            predictions = net(images.to(device))
            loss = F.cross_entropy(predictions, labels.to(device))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            batch_correct = get_num_correct(predictions, labels.to(device))
            total_correct += batch_correct
            total_loss += loss.item()

            # Print batch progress
            batch_accuracy = 100 * batch_correct / len(labels)
            print(f'Batch {i + 1}/{len(train_loader)}, Batch Accuracy: {batch_accuracy:.2f}%')

        # Epoch summary
        accuracy = 100 * total_correct / len(train_dataset)
        best_accuracy = max(best_accuracy, accuracy)
        print(f'Epoch: {epoch + 1}, Total Correct: {total_correct}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

        epoch += 1

    return best_accuracy


# Train the model
print("\nStarting training...")
final_accuracy = train_model(epochs=10, target_accuracy=90)
print(f"Training complete. Final accuracy: {final_accuracy:.2f}%")

# Save the model
model_path = os.path.join(BASE_DIR, "modelv5.pt")
torch.save(net.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load test dataset
transform_test = transforms.Compose([
    transforms.CenterCrop((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("\nLoading test dataset...")
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform_test)
original_test_labels_dict = test_dataset.class_to_idx
print(f"Found {len(test_dataset)} test images in {len(original_test_labels_dict)} classes")

# Load test JSON labels
print("Loading test JSON labels...")
with open(TEST_JSON, 'r') as f:
    label_test = json.load(f)

# Create test label mapping using the same approach as for training
test_label_map_dict = {}
for folder, idx in original_test_labels_dict.items():
    # Try different key formats to find a match
    folder_key = folder

    # Method 1: Direct match
    if folder_key in label_test:
        test_label_map_dict[idx] = label_test[folder_key]
        continue

    # Method 2: Check if folder name is in any JSON key
    found = False
    for json_key in label_test.keys():
        # Extract the folder name from JSON keys
        json_folder = json_key.split('/')[-1] if '/' in json_key else json_key
        if folder_key == json_folder:
            test_label_map_dict[idx] = label_test[json_key]
            found = True
            break

    # Method 3: Try fuzzy matching
    if not found:
        for json_key in label_test.keys():
            if folder_key in json_key or folder_key.replace(" ", "") in json_key.replace("/", ""):
                test_label_map_dict[idx] = label_test[json_key]
                found = True
                break

    if not found:
        print(f"WARNING: No matching test label found for folder '{folder_key}'")
        test_label_map_dict[idx] = -1

print("\nTest label mapping:")
print(test_label_map_dict)

# Handle missing test mappings the same way as for training
if -1 in test_label_map_dict.values():
    missing_folders = [folder for folder, idx in original_test_labels_dict.items()
                       if test_label_map_dict[idx] == -1]
    print(f"ERROR: Missing test labels for folders: {missing_folders}")

    # Print possible matches to help debugging
    print("\nPossible JSON keys that might match:")
    for missing_folder in missing_folders:
        similar_keys = [k for k in label_test.keys() if any(part in k for part in missing_folder.split())]
        print(f"  For '{missing_folder}', possible matches: {similar_keys}")

    # Manual mapping for test folders
    manual_test_mapping = {
        # Example: 'folder_name': corresponding_label_value
        # '170815-133921-Al 2mm': 0,  # Adjust this based on your data
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

# Test the model
torch.set_grad_enabled(False)
test_loader = DataLoader(test_dataset, batch_size=200)

print("\nEvaluating on test set...")
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
    batch_accuracy = 100 * total_correct / total
    print(f'Test Batch {i + 1}/{len(test_loader)}, Accuracy: {batch_accuracy:.2f}%')

# Final test accuracy
test_accuracy = 100 * total_correct / len(test_dataset)
print(f'\nTest Evaluation Complete. Final Test Accuracy: {test_accuracy:.2f}%')
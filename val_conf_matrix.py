import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

# Load DINO model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define the model architecture
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Adjust the number of classes as needed
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

# Load the model
def load_model(model_path, device):
    model = DinoVisionTransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model.to(device)

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((280, 280)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization for pre-trained models
])

# Load dataset and create DataLoader
val_dataset = ImageFolder(root='Dataset-challnge/test', transform=data_transforms)  # Adjust path
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = "models/dino_vit_classifier_capsule.pth"
model = load_model(model_save_path, device)

# Set to evaluation mode
model.eval()

# Prepare for validation set evaluation
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in val_loader:  # Use val_loader defined above
        images, labels = data
        all_labels.append(labels.cpu().numpy())  # Store true labels
        
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.append(predicted.cpu().numpy())  # Store predictions

# Flatten lists of predictions and labels
all_predictions = np.concatenate(all_predictions)
all_labels = np.concatenate(all_labels)

# Generate and plot the normalized confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Generate classification report
class_report = classification_report(all_labels, all_predictions, target_names=val_dataset.classes)
print("Classification Report:\n", class_report)

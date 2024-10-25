import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# load dino model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define the model architecture
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14  # Assuming dinov2_vits14 is already defined elsewhere
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

# Define the image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(280),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load and process the validation images
def load_validation_images(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
    return image_paths

# Make predictions
def make_predictions(model, image_paths, device):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
                     'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    
    all_image_paths = []
    all_predictions = []
    all_labels = []  # Assuming you have labels for validation data
    
    data_transforms = get_transforms()

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            input_tensor = data_transforms(image).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_class = np.argmax(probabilities, axis=1)

            all_image_paths.append(os.path.basename(img_path))
            all_predictions.append(probabilities[0])
            
            # Here, you should add the actual label for each image
            # This is a placeholder; replace with actual logic to get the label
            label = get_label_for_image(img_path)  # Implement this function based on your dataset
            all_labels.append(label)

    return all_labels, np.array(all_predictions), class_columns

# Save predictions in the desired format
def save_predictions_to_excel(image_paths, predictions, class_columns, output_path):
    df_prob = pd.DataFrame(predictions, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    
    y_pred_classes = np.argmax(predictions, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df_prob['predicted_class'] = predicted_class_names
    
    df_prob.to_excel(output_path, index=False)  # Save as Excel file

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix (Without Normalization)")
    
    print(cm)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def generate_classification_report_and_confusion_matrix(y_true, y_pred_proba, class_columns, normalize_cm=False):
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    class_report = classification_report(y_true, y_pred, target_names=class_columns, zero_division=0)
    print("Classification Report:\n", class_report)

    plot_confusion_matrix(y_true, y_pred, class_columns, normalize=normalize_cm)

# Placeholder function to get the label for an image
def get_label_for_image(image_path):
    # Get the directory name (class label) from the image path
    class_label = os.path.basename(os.path.dirname(image_path))
    
    # Define a mapping from class labels to integer values
    label_map = {
        'Angioectasia': 0,
        'Bleeding': 1,
        'Erosion': 2,
        'Erythema': 3,
        'Foreign Body': 4,
        'Lymphangiectasia': 5,
        'Normal': 6,
        'Polyp': 7,
        'Ulcer': 8,
        'Worms': 9
    }
    
    # Return the mapped label, or -1 if the label is not found
    return label_map.get(class_label, -1)

if __name__ == "__main__":
    # Define paths
    model_save_path = "models/dino_vit_classifier_capsule.pth"
    validation_image_folder = 'Dataset-challnge/test'
    output_path = 'submission_val_predictions.xlsx'  # Change to .xlsx

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_save_path, device)

    # Load validation images
    image_paths = load_validation_images(validation_image_folder)

    # Make predictions
    all_labels, all_predictions, class_columns = make_predictions(model, image_paths, device)

    # Save predictions to Excel
    save_predictions_to_excel(image_paths, all_predictions, class_columns, output_path)

    # Generate classification report and confusion matrix
    generate_classification_report_and_confusion_matrix(all_labels, all_predictions, class_columns, normalize_cm=True)

    print(f"Predictions and report saved to {output_path}")

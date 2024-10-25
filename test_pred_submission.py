import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn

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

# Load and process the test images
def load_test_images(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
    return image_paths

# Make predictions
def make_predictions(model, image_paths, device):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
                     'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    
    all_image_paths = []
    all_predictions = []
    
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

    return all_image_paths, np.array(all_predictions), class_columns

# Save predictions in the desired format (Excel)
def save_predictions_to_excel(image_paths, predictions, class_columns, output_path):
    df_prob = pd.DataFrame(predictions, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    
    y_pred_classes = np.argmax(predictions, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df_prob['predicted_class'] = predicted_class_names
    
    df_prob.to_excel(output_path, index=False, engine='openpyxl')

if __name__ == "__main__":
    # Define paths
    model_save_path = "models/dino_vit_classifier_capsule.pth"
    test_image_folder = 'Testing set/Testing set/Images'
    output_path = 'submission_capsule.xlsx'  # Changed to .xlsx

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_save_path, device)

    # Load test images
    image_paths = load_test_images(test_image_folder)

    # Make predictions
    all_image_paths, all_predictions, class_columns = make_predictions(model, image_paths, device)

    # Save predictions to Excel
    save_predictions_to_excel(all_image_paths, all_predictions, class_columns, output_path)

    print(f"Predictions saved to {output_path}")

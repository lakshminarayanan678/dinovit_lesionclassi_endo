import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import random
import numpy as np
import wandb


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(100)

    wandb.init(
        project="capsule_vision_challenge_2024",  # Replace with your W&B project
        name="dinovit_internal_split_run",          # Optional run name
        config={
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.000001,
            "model": "dinovit_internal_split"
        }
    )

    # Create dummy dataset artifact
    dataset_artifact = wandb.Artifact('ue_internal_split', type='dataset')
    dataset_artifact.add_dir('/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data/train_val')
    wandb.log_artifact(dataset_artifact)


    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(280),
            transforms.RandomResizedCrop(280),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(280),
            transforms.CenterCrop(280),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_dinovit/train_val'  # Update the path as needed
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load DINO model
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = dinov2_vits14
            self.classifier = nn.Sequential(
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, 3)  # Change to the number of classes in your dataset
            )
        
        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
            return x

    # Initialize model
    model = DinoVisionTransformerClassifier()
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # Training loop
    for epoch in range(50):  # Adjust the number of epochs as needed
        running_loss = 0.0
        for i, data in enumerate(dataloaders["train"], 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:  # Print every 50 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the model
    model_save_path = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/codes/Capsule-Challenge-2024/models/OURS_Split_50epochs_dino_vit_classifier_capsule.pth"
    torch.save(model.state_dict(), model_save_path)

    # Log model weights as artifact
    model_artifact = wandb.Artifact('dinovit-internal-split', type='model')
    model_artifact.add_file(model_save_path)
    wandb.log_artifact(model_artifact)

    print(f"Model saved to {model_save_path}")
    wandb.finish()

if __name__ == '__main__':
    main()
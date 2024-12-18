Here's the updated README with information about the 1st rank and the citation:

---

# DinoVision Transformer Classifier

## Overview

The DinoVision Transformer Classifier is a machine learning project that utilizes the Vision Transformer architecture to classify images into different medical categories. The project is built using PyTorch and leverages transfer learning techniques with the DinoV2 model for feature extraction and classification tasks. 

**This project achieved 1st place in the Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy, hosted by CVIP 2024, organized by Danube Private University Austria (DPU), @CVIP 2024 @MISAHUB 🏢, and @Indian Institute of Information Technology, Design and Manufacturing, Jabalpur.**

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Project Structure

```
DinoVisionTransformerClassifier/
├── models/
│   ├── dino_vit_classifier_capsule.pth  # Trained model weights
├── Dataset-challnge/
│   └── test/
│       ├── Angioectasia/
│       ├── Bleeding/
│       ├── Erosion/
│       ├── Erythema/
│       ├── Foreign Body/
│       ├── Lymphangiectasia/
│       ├── Normal/
│       ├── Polyp/
│       ├── Ulcer/
│       └── Worms/
├── train_model.py                          # To train the model
├── test_pred_submission.py                 # To make predictions on the validation set and save the results
├── val_conf_matrix.py                      # To visualize the performance of the model using a confusion matrix
├── val_scores.py                           # To compute and display various validation scores, including accuracy and F1-score
├── requirements.txt                        # Python package dependencies
└── README.md                               # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sri-Karthik-Avala/Capsule-Challenge-2024.git
   cd Capsule-Challenge-2024
   ```

2. **Install required packages**:
   Create a virtual environment and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Setup the Environment
Before running the scripts, ensure that you have set up the environment as described in the [Installation](#installation) section.

### 2. Model Training and Prediction

#### Step 1: Train the Model
To train the model, run the following command:
```bash
python train_model.py
```
This script will:
- Load the training dataset and apply the necessary transformations.
- Train the DinoVisionTransformer model.
- Save the trained model weights to a specified location (ensure to update the path in the script as necessary).

#### Step 2: Validate and Make Predictions
To make predictions on the validation set and save the results, execute:
```bash
python test_pred_submission.py
```
This script will:
- Load the pre-trained model from the specified path.
- Load the validation images from the designated folder structure (similar to the training step).
- Generate predictions for each image and save the results in a CSV file named `submission_val_predictions.csv`.

#### Step 3: Generate Confusion Matrix
To visualize the performance of the model using a confusion matrix, run:
```bash
python val_conf_matrix.py
```
This script will:
- Load the predictions and true labels.
- Plot and display the confusion matrix, allowing you to see how well the model is performing across different classes.

#### Step 4: Calculate Validation Scores
To compute and display various validation scores, including accuracy and F1-score, execute:
```bash
python val_scores.py
```
This script will:
- Load the true labels and predicted probabilities.
- Generate a classification report that summarizes precision, recall, and F1 scores for each class.


## Model Architecture

The model consists of:
- A **Vision Transformer** (ViT) as the feature extractor, specifically using the DinoV2 architecture.
- A fully connected classification head with two layers:
  - **Linear Layer**: Maps the output of the transformer to a lower-dimensional space (256 neurons).
  - **ReLU Activation**: Introduces non-linearity.
  - **Final Linear Layer**: Outputs the class probabilities (10 classes).

## Data Preparation

The dataset is organized into class folders representing different medical conditions. The model is trained to classify images based on the folder structure, where each folder name corresponds to a specific class label.

## Training and Evaluation

1. **Training**: The model is pre-trained on a dataset using transfer learning techniques.
2. **Evaluation**: The model's performance is evaluated using:
   - Confusion Matrix: Visualizes the performance across different classes.
   - Classification Report: Provides precision, recall, F1-score, and support for each class.

## Results

The results of the predictions are saved in a CSV file, which includes:
- Image paths
- Predicted probabilities for each class
- Final predicted class label

Additionally, classification reports and confusion matrices are displayed for further analysis.

## Contributing

Contributions are welcome! If you have suggestions for improvements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Citation

If you use this work, please cite it as:

```
@misc{handa2024capsulevision2024challenge,
      title={Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy}, 
      author={Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Manas Dhir and Deepti Chhabra and Shreshtha Jha and Pallavi Sharma and Vijay Thakur and Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Raman},
      year={2024},
      eprint={2408.04940},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04940}, 
}
``` 

---

Let me know if you'd like further modifications!

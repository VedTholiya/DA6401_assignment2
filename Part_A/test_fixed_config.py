import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
from n1 import CNN

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test data path
test_data_path = "C:/Users/vedpr/OneDrive/Documents/GitHub/DA6401_assignment2/Part_A/nature_12K/inaturalist_12K/val"

# Transform for test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load test data
test_data = ImageFolder(root=test_data_path, transform=transform)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
classes = test_dataloader.dataset.classes

# Model configuration
config = {
    'number_of_filters': 64,
    'filter_size': 3,
    'stride': 1,
    'padding': 1,
    'max_pooling_size': 2,
    'n_neurons': 64,
    'n_classes': 10,
    'conv_activation': 'silu',
    'dense_activation': 'sigmoid',
    'dropout_rate': 0.2,
    'use_batchnorm': True,
    'factor': 3,
    'dropout_organisation': 2,
}

# Initialize model
model = CNN(
    input_dimension=(3, 224, 224),
    number_of_filters=config['number_of_filters'],
    filter_size=(config['filter_size'], config['filter_size']),
    stride=config['stride'],
    padding=config['padding'],
    max_pooling_size=(config['max_pooling_size'], config['max_pooling_size']),
    n_neurons=config['n_neurons'],
    n_classes=config['n_classes'],
    conv_activation=nn.ReLU(),
    dense_activation=nn.ReLU6(),
    dropout_rate=config['dropout_rate'],
    use_batchnorm=config['use_batchnorm'],
    factor=config['factor'],
    dropout_organisation=config['dropout_organisation']
).to(device)

# Load trained model
model.load_state_dict(torch.load("best_model_fixed_config.pth", map_location=device))
model.eval()

def predict():
    y_pred = []
    y_true = []

    # Iterate over test data
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate accuracy
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_predictions / len(y_true)
    print(f"Overall accuracy: {accuracy:.4f}")

    # Create confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=classes,
        columns=classes
    )

    # Plot confusion matrix
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_fixed_config.png')
    plt.close()

if __name__ == "__main__":
    predict()
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
import wandb

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

# Model configuration - using the same configuration as best training config from sweep
config = {
    'number_of_filters': 128,  
    'filter_size': 3,
    'stride': 1,
    'padding': 1,
    'max_pooling_size': 2,
    'n_neurons': 128,  
    'n_classes': 10,
    'conv_activation': 'relu',
    'dense_activation': 'relu6',
    'dropout_rate': 0.2,
    'use_batchnorm': True,
    'factor': 0.5,  
    'dropout_organisation': 3,
}

# Initialize wandb
wandb.login(key='f15dba29e56f32e9c31d598bce5bc7a3c76de62e')
wandb.init(project="DA6401_Assignment2", config=config, name="sweep_best_test")

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

# Load the best model from sweep
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

def create_prediction_grid():
    model.eval()
    # Get 30 random samples (10x3 grid)
    all_indices = list(range(len(test_data)))
    sample_indices = np.random.choice(all_indices, 30, replace=False)
    
    fig, axes = plt.subplots(10, 3, figsize=(15, 40))
    fig.suptitle('Sample Predictions (True vs Predicted)', fontsize=16)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            row = idx // 3
            col = idx % 3
            
            # Get image and label
            image, label = test_data[sample_idx]
            true_label = classes[label]
            
            # Get prediction
            output = model(image.unsqueeze(0).to(device))
            _, predicted = torch.max(output, 1)
            pred_label = classes[predicted.item()]
            
            # Convert tensor to image
            img = image.permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = img.clip(0, 1)
            
            # Plot
            axes[row, col].imshow(img)
            color = 'green' if true_label == pred_label else 'red'
            axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    wandb.log({"prediction_grid": wandb.Image(plt)})
    plt.close()

def predict():
    y_pred = []
    y_true = []
    test_loss = []
    criterion = nn.CrossEntropyLoss()

    # Iterate over test data
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate accuracy
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_predictions / len(y_true)
    avg_loss = sum(test_loss) / len(test_loss)
    print(f"Overall accuracy: {accuracy:.4f}")

    # Add prediction grid
    create_prediction_grid()

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
    plt.title('Confusion Matrix_best_sweep')
    plt.savefig('confusion_matrix_sweep_best.png')

    # Log to wandb
    wandb.log({
        "test_accuracy": accuracy,
        "test_loss": avg_loss,
        "confusion_matrix": wandb.Image(plt)
    })
    plt.close()

    # Create accuracy vs loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_loss, label='Loss')
    plt.axhline(y=accuracy, color='r', linestyle='-', label='Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Value')
    plt.title('Test Loss vs Accuracy')
    plt.legend()
    wandb.log({"loss_vs_accuracy": wandb.Image(plt)})
    plt.close()

if __name__ == "__main__":
    predict()
    wandb.finish()
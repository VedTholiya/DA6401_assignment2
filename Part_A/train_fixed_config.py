import gc
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from n1 import CNN
import matplotlib.pyplot as plt
import wandb

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
training_data_path = "C:/Users/vedpr/OneDrive/Documents/GitHub/DA6401_assignment2/Part_A/nature_12K/inaturalist_12K/train"

# Fixed configurations
config = {
    'number_of_filters': 32,
    'filter_size': 3,
    'stride': 1,
    'padding': 1,
    'max_pooling_size': 2,
    'n_neurons': 512,
    'n_classes': 10,
    'conv_activation': 'relu',
    'dense_activation': 'relu6',
    'dropout_rate': 0.2,
    'use_batchnorm': True,
    'factor': 1,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'epochs': 100,
    'use_augmentation': False,
    'dropout_organisation': 3,
}

# Initialize wandb
wandb.login(key='f15dba29e56f32e9c31d598bce5bc7a3c76de62e')
wandb.init(project="DA6401_Assignment2", config=config, name="fixed_config_run")

def get_transform(use_augmentation):
    if use_augmentation:
        return transforms.Compose([
            transforms.RandomCrop(50, padding=1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 20)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

def train_model():
    # Load and split dataset
    dataset = ImageFolder(root=training_data_path, transform=get_transform(config['use_augmentation']))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

    # Setup activations
    activations = {
        'relu': nn.ReLU(),
        'relu6': nn.ReLU6(),
    }

    # Initialize model
    gc.collect()
    torch.cuda.empty_cache()
    model = CNN(
        input_dimension=(3, 224, 224),
        number_of_filters=config['number_of_filters'],
        filter_size=(config['filter_size'], config['filter_size']),
        stride=config['stride'],
        padding=config['padding'],
        max_pooling_size=(config['max_pooling_size'], config['max_pooling_size']),
        n_neurons=config['n_neurons'],
        n_classes=config['n_classes'],
        conv_activation=activations[config['conv_activation']],
        dense_activation=activations[config['dense_activation']],
        dropout_rate=config['dropout_rate'],
        use_batchnorm=config['use_batchnorm'],
        factor=config['factor'],
        dropout_organisation=config['dropout_organisation'],
    ).to(device)

    # Training setup
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
            del x, y
            
        train_accuracy = 100 * correct / total
        avg_train_loss = train_loss / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
                del x, y
                
        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        wandb.log({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })

        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        print("-" * 50)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model_fixed_config.pth")

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("Training history plot saved as 'training_history.png'")
    
    wandb.finish()
    return model

if __name__ == "__main__":
    model = train_model()
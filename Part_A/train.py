import gc
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from n1 import CNN
import wandb
from torch.cuda.amp import GradScaler, autocast

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable deterministic training for reproducibility
torch.backends.cudnn.benchmark = True

# Dataset path
training_data_path = "C:/Users/vedpr/OneDrive/Documents/GitHub/DA6401_assignment2/Part_A/nature_12K/inaturalist_12K/train"

# Fixed hyperparameters
config = {
    'number_of_filters': 64,
    'filter_size': 3,
    'stride': 1,
    'padding': 1,
    'max_pooling_size': 2,
    'n_neurons': 256,
    'n_classes': 10,
    'conv_activation': 'relu',
    'dense_activation': 'relu',
    'dropout_rate': 0.3,
    'use_batchnorm': True,
    'factor': 2,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'epochs': 10,
    'use_augmentation': True,
    'dropout_organisation': 3
}

sweep_config = {
    'method': 'bayes',
    'name': 'PART_A_Q2_SWEEP_1',
    'metric': {'name': "val_accuracy", 'goal': 'maximize'},
    'parameters': {
        'number_of_filters': {'values': [16, 32, 64, 128, 256]},
        'filter_size': {'value': 3},
        'stride': {'value': 1},
        'padding': {'value': 1},
        'max_pooling_size': {'value': 2},
        'n_neurons': {'values': [64, 128, 256, 512, 1024]},
        'n_classes': {'value': 10},
        'conv_activation': {'values': ['relu', 'gelu', 'silu', 'mish', 'relu6', 'tanh', 'sigmoid']},
        'dense_activation': {'values': ['relu', 'gelu', 'silu', 'mish', 'relu6', 'tanh', 'sigmoid']},
        'dropout_rate': {'values': [0.2, 0.3, 0.4, 0.5]},
        'use_batchnorm': {'values': [True, False]},
        'factor': {'values': [1, 2, 3, 0.5]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
        'batch_size': {'value': 16},
        'epochs': {'values': [5, 10, 15]},
        'use_augmentation': {'values': [True, False]},
        'dropout_organisation': {'values': [1, 2, 3, 4, 5]},
    },
}

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
    # Dataset
    dataset = ImageFolder(root=training_data_path, transform=get_transform(config['use_augmentation']))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Activation options
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
        'relu6': nn.ReLU6(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }

    # Model
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

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
            del x, y

        train_accuracy = 100 * correct / total
        avg_train_loss = train_loss / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
                del x, y

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / total

        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        print("-" * 50)

    # Save the trained model
    save_path = "saved_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model

def wandb_sweep():
    with wandb.init() as run:
        sweep_config = dict(wandb.config)
        dataset = ImageFolder(root=training_data_path, transform=get_transform(sweep_config['use_augmentation']))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        # Optimized DataLoaders
        train_loader = DataLoader(
            train_set, 
            batch_size=sweep_config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=sweep_config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'relu6': nn.ReLU6(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        gc.collect()
        torch.cuda.empty_cache()
        model = CNN(
            input_dimension=(3, 224, 224),
            number_of_filters=sweep_config['number_of_filters'],
            filter_size=(sweep_config['filter_size'], sweep_config['filter_size']),
            stride=sweep_config['stride'],
            padding=sweep_config['padding'],
            max_pooling_size=(sweep_config['max_pooling_size'], sweep_config['max_pooling_size']),
            n_neurons=sweep_config['n_neurons'],
            n_classes=sweep_config['n_classes'],
            conv_activation=activations[sweep_config['conv_activation']],
            dense_activation=activations[sweep_config['dense_activation']],
            dropout_rate=sweep_config['dropout_rate'],
            use_batchnorm=sweep_config['use_batchnorm'],
            factor=sweep_config['factor'],
            dropout_organisation=sweep_config['dropout_organisation'],
        ).to(device)
        optimizer = Adam(model.parameters(), lr=sweep_config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        best_val_accuracy = 0.0
        for epoch in range(sweep_config["epochs"]):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                
                # Mixed precision training
                with autocast():
                    pred = model(x)
                    loss = criterion(pred, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * x.size(0)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
                del x, y
            train_accuracy = 100 * correct / total
            avg_train_loss = train_loss / total
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item() * x.size(0)
                    correct += (pred.argmax(1) == y).sum().item()
                    total += y.size(0)
                    del x, y
            val_accuracy = 100 * correct / total
            avg_val_loss = val_loss / total
            wandb.log({
                'epoch': epoch+1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            })
            print(f"Epoch [{epoch+1}/{sweep_config['epochs']}]\nTrain Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%\nVal Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n" + "-" * 50)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved to best_model.pth with val_accuracy={best_val_accuracy:.2f}%")

if __name__ == "__main__":
    wandb.login(key='f15dba29e56f32e9c31d598bce5bc7a3c76de62e')
    sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment2")
    wandb.agent(sweep_id, function=wandb_sweep, count=3)  
    wandb.finish()
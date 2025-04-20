import argparse
import torch
import wandb
from n1 import CNN
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

WANDB_PROJECT = "DA6401_Assignment2"
WANDB_ENTITY = "ma23c047-indian-institute-of-technology-madras"

def get_transform(use_augmentation):
    if use_augmentation:
        return transforms.Compose([
            transforms.RandomCrop(50, padding=1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 20)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_data = ImageFolder(
        root=training_data_path, transform=get_transform(config['use_augmentation'])
    )
    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size
    train_set, validation_set = random_split(training_data, [train_size, val_size])
    
    train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False)

    activations = {
        "relu": nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
        "relu6": nn.ReLU6(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }

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

    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        model.train()
        running_loss = running_accuracy = n_samples = 0
        
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            loss = criterion(pred, y)
            
            running_loss += loss.item() * x.size(0)
            running_accuracy += (pred.argmax(1) == y).sum().item()
            n_samples += y.size(0)
            
            loss.backward()
            optimizer.step()

        train_loss = running_loss / n_samples
        train_accuracy = 100 * running_accuracy / n_samples

        model.eval()
        running_loss = running_accuracy = n_samples = 0
        
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                
                running_loss += loss.item() * x.size(0)
                running_accuracy += (pred.argmax(1) == y).sum().item()
                n_samples += y.size(0)

        val_loss = running_loss / n_samples
        val_accuracy = 100 * running_accuracy / n_samples

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("-we", "--wandb_entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-nf", "--number_of_filters", type=int, default=16)
    parser.add_argument("-fs", "--filter_size", type=int, default=3)
    parser.add_argument("-ps", "--max_pooling_size", type=int, default=2)
    parser.add_argument("-s", "--stride", type=int, default=1)
    parser.add_argument("-p", "--padding", type=int, default=1)
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.2)
    parser.add_argument("-ca", "--conv_activation", type=str, default="relu")
    parser.add_argument("-da", "--dense_activation", type=str, default="relu6")
    parser.add_argument("-nn", "--n_neurons", type=int, default=512)
    parser.add_argument("-nc", "--n_classes", type=int, default=10)
    parser.add_argument("-f", "--factor", type=float, default=1)
    parser.add_argument("-do", "--dropout_organisation", type=int, default=3)
    parser.add_argument("-bn", "--use_batchnorm", type=bool, default=True)
    parser.add_argument("-aug", "--use_augmentation", type=bool, default=False)

    args = parser.parse_args()
    config = vars(args)

    wandb.init(
        project=config['wandb_project'],
        entity=config['wandb_entity'],
        config=config
    )

    training_data_path = "C:\Users\vedpr\OneDrive\Documents\GitHub\DA6401_assignment2\Part_A\nature_12K\inaturalist_12K\train"  
    model = train(config)
    
    torch.save(model.state_dict(), "model_a.pth")
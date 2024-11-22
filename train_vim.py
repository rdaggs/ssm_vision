import os
import sys
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.cuda.amp import autocast, GradScaler

def train_model(model,train_loader,criterion,optimizer,num_epochs,device,exp):
    model.train()
    print('training')

    for e in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0

        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss+=loss.item()
            print(running_loss)

        # print loss
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time  # Calculate epoch duration
        print(f"Epoch [{e + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # save model

    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': e,
                  'loss': running_loss,
                 }

    torch.save(checkpoint, f'vim_checkpoint_{num_epochs}.pth')


def evaluate_model(model,test_loader,device):
    model.eval()
    print('testing')
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy



if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir)
    os.chdir(relative_path)
    print(f"Changed working directory to: {os.getcwd()}")

    # Set up data transforms and loaders
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    from vim import Vim
    model = Vim(
        dim=256,
        dt_rank=32,
        dim_inner=256,
        d_state=256,
        num_classes=10,
        image_size=32,
        patch_size=16,
        channels=3,
        dropout=0.1,
        depth=4,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Experiment name
    exp ='pretrained_for_pruning_technique'

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, num_epochs=100, device=device, exp=exp)
    evaluate_model(model, test_loader, device)
import os
import sys
import math
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
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
from vim import Vim


def train_model(model, train_loader, test_loader, criterion, optimizer,scheduler, num_epochs, device, experiment, checkpoint_path=None):
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print(f"iniate training process")

    # print(optimizer.param_groups[0]['lr'])
    model.train()
    start_time = time.time()

    for e in range(start_epoch, num_epochs):
        start_time = time.time()
        running_loss = 0.0
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # capping norm of gradients to 1, stabilizing training with high LR 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            if math.isnan(running_loss):
                print("nan detected in running loss. stopping training.")
                break
        epoch_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)  # possibly include penalty term 
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"epoch [{e + 1}/{num_epochs}], training loss: {epoch_loss:.4f}, validation loss: {val_loss:.4f}, time: {epoch_time:.2f}s, learning rate: {current_lr:.6f}")

        # intermediate checkpoint
        if (e + 1) % 5 == 0:
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': e,
                          'loss': epoch_loss}
            torch.save(checkpoint, f'checkpoints/{experiment}_epoch_{e + 1}.pth')
    
    # save final checkpoint
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': num_epochs - 1,
                  'loss': epoch_loss}
    torch.save(checkpoint, f'checkpoints/{experiment}_final.pth')


def evaluate_model(model,test_loader,criterion,device,checkpoint_path):
    if checkpoint_path:
          checkpoint = torch.load(checkpoint_path, map_location=device)
          model.load_state_dict(checkpoint['model_state_dict'])
          print(f"Loaded model from checkpoint: {checkpoint_path}")
    model.eval()
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
    print(f"Accuracy: {accuracy:.4f}% Test Loss: {avg_loss:.4f}, ")
    return accuracy


def calculate_model_sparsity(model):
    
    total_weights,zero_weights = 0,0
    
    for name, layer in model.named_modules():
        if hasattr(layer,'weight'):     # ensure layer has weight attribute
            weights = layer.weight
            zero_weights += torch.sum(weights == 0).sum().item()
            total_weights+= torch.numel(weights)
    
    sparsity = (100 * zero_weights / total_weights)  if total_weights > 0 else 0
    return sparsity

def calculate_mask_sparsity(mask):
    total_elements = 0
    zero_elements = 0

    for name, tensor in mask.items():
        total_elements += tensor.numel()  # total elements in the mask
        zero_elements += (tensor == 0).sum().item()  # zero elements

    sparsity = 100.0 * zero_elements / total_elements if total_elements > 0 else 0
    return sparsity


def pruning_binary_mask(model,prune_intensity,mask):
    assert mask != None 

    for layer_name, layer in model.named_modules():
        for param_name, parameter in layer.named_parameters(recurse=False):
            full_name = f"{layer_name}.{param_name}" if layer_name else param_name
            
            if full_name in mask:
                weight = parameter.data
                curr_mask = mask[full_name]
                
                # nonzero weights that are currently masked in
                masked_weights = weight * curr_mask
                nonzero_weights = masked_weights[masked_weights != 0]
                
                if len(nonzero_weights) > 0:  # Check if there are any nonzero weights
                    # calculate threshold only accounting for nonzero weights
                    threshold = torch.quantile(torch.abs(nonzero_weights), prune_intensity)
                    
                    #  weights that are both nonzero in masked_weights / bove the threshold
                    new_mask = (torch.abs(masked_weights) >= threshold).float()
                    new_mask *= (masked_weights != 0).float()
                    
                    # Update the mask and weights
                    mask[full_name] *= new_mask
                    parameter.data *= mask[full_name]
    
    return model, mask


def iterative_magnitude_pruning(prune_intensity, prune_iterations,finetuning_epochs, model, train_loader, test_loader, criterion, optimizer,scheduler, device, experiment, checkpoint_path):
    cumulative_mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if "weight" in name}
    initial_mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if "weight" in name}
    pretrained = torch.load(checkpoint_path, map_location=device)

    for p in range(prune_iterations):
        print(f'pruning iteration {p}')

        prune_intensity_update = prune_intensity + p * 0.04

        model.load_state_dict(pretrained['model_state_dict'])
        #========================================================================#
        epoch_fix = pretrained['epoch'] + 1 + finetuning_epochs
        #========================================================================#
        train_model(model, train_loader, test_loader, criterion, optimizer,scheduler, epoch_fix, device, experiment, checkpoint_path)
        _, pruned_mask = pruning_binary_mask(model, prune_intensity_update, cumulative_mask)
        
        for name in cumulative_mask:
            cumulative_mask[name] = torch.logical_and(cumulative_mask[name].bool(), pruned_mask[name].bool()).float()
        
        print(f"cumulative mask sparsity at pruning iteration {p}: {calculate_mask_sparsity(cumulative_mask):.4f}%")
        
        accuracy = evaluate_model(model,test_loader,criterion,device,checkpoint_path)
        
        prune_checkpoint = {'model_state_dict': model.state_dict()}
        torch.save(prune_checkpoint, f'checkpoints/{experiment}_iter_{p+1}.pth')
        torch.save(cumulative_mask,  f'checkpoints/{experiment}_iter_{p+1}_mask.pth')

        print('checkpoint/mask saved')
    
    return cumulative_mask, accuracy



if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir)
    os.chdir(relative_path)
    print(f"Changed working directory to: {os.getcwd()}")

    test_epochs = 2
    batch_size = 8
    num_epochs = test_epochs

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
                depth=10,
                )

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    print('initializing data')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=0.001, total_iters=num_epochs, last_epoch=-1, verbose=True)

    train_model(model,train_loader,test_loader, criterion, optimizer,scheduler,7,device,'testing_vim_training',checkpoint_path='checkpoints/deep_vim__150_epoch_5.pth')




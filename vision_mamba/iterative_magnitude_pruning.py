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
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts, CyclicLR
from torchinfo import summary
from torchvision.datasets import CIFAR100
import torch.nn.utils.prune as prune
from train_eval import train_model, evaluate_model  
from vim import Vim



def calculate_model_sparsity(model):
    
    total_weights,zero_weights = 0,0
    
    for name, layer in model.named_modules():
        if hasattr(layer,'weight'):
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



def apply_pruning(model, sparsity):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)
            prune.remove(module, "weight")


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
                    
                    mask[full_name] *= new_mask
                    parameter.data *= mask[full_name]
    
    return model, mask


def iterative_magnitude_pruning(prune_intensity, prune_iterations,finetuning_epochs, model, train_loader, test_loader, criterion, optimizer,scheduler, device, experiment, checkpoint_path):
    cumulative_mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if "weight" in name}
    initial_mask = {name: torch.ones_like(param) for name, param in model.named_parameters() if "weight" in name}
    pretrained = torch.load(checkpoint_path, map_location=device)

    for p in range(2,prune_iterations):
        print(f'pruning iteration {p}')

        prune_intensity_update = prune_intensity + p * 0.04

        model.load_state_dict(pretrained['model_state_dict'])
        #========================================================================#
        epoch_fix = pretrained['epoch'] + 1+ finetuning_epochs
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





def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir)
    os.chdir(relative_path)
    print(f"Changed working directory to: {os.getcwd()}")

    num_epochs = 150
    batch_size = 8

    
    model = Vim(dim=256,
                dt_rank=32,
                dim_inner=256,
                d_state=256,
                num_classes=10,
                image_size=32,
                patch_size=16,
                channels=3,
                dropout=0.1,
                depth=10,)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode = 'min',patience=8, factor=0.1,verbose=True)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult=2)
    scheduler = CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-4, step_size_up=2000, mode="triangular")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    exp ='pretrained_for_pruning_technique'

    train_model(model, train_loader, criterion, optimizer, scheduler, 150, device,exp)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()

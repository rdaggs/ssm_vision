import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vim import Vim


def find_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            layers[name] = module
    return layers

def compute_gradients(model, dataloader, device, scale):
    gradients_l1,gradients_l2 = {},{}
    for nm, param in find_layers(model).items():
        gradients_l1[nm]=torch.zeros_like(param.weight, dtype=torch.float16, device=device)
        gradients_l2[nm]=torch.zeros_like(param.weight, dtype=torch.float32, device=device)
    nsamples = 0
    model.train()
    for inputs, labels in tqdm(dataloader, desc="batch processing"):
        nsamples += 1
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        for nm, layer in find_layers(model).items():
            
            if layer.weight.grad is not None:
                grad = layer.weight.grad.detach().clone().to(dtype=torch.float32)
                gradients_l1[nm] += torch.abs(grad * scale).to(device).to(dtype=torch.float16)
                gradients_l2[nm] += (grad*scale).pow(2).to(device)
        
        model.zero_grad()
    
    for nm in gradients_l2:
        gradients_l2[nm] = torch.sqrt(gradients_l2[nm].to(dtype=torch.float16))
    
    return gradients_l1, gradients_l2

def save_gradients(gradients, exp, model):
    gradient_path = f'./gradients/{exp}'
    os.makedirs(gradient_path, exist_ok=True)
    l2_path = os.path.join(gradient_path, f"/checkpointsgrads_l2_{model}.pth")
    l1_path = os.path.join(gradient_path, f"/checkpoints/grads_l1_{model}.pth")
    torch.save(gradients[1], l2_path)
    torch.save(gradients[0], l1_path)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir)
    os.chdir(relative_path)
    print(f"Changed working directory to: {os.getcwd()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

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

    model.train()
    scale = [10,20,30,40,50,100]
    for s in scale:
        gradients = compute_gradients(model, train_loader, device, scale)
        save_gradients(gradients, s, f'grad_pruning_test_scale_{s}')
    

if __name__ == "__main__":
    main()
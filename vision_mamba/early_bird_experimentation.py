import torch
import torch.nn.utils.prune as prune
from collections import deque

from train_eval import train_model, evaluate_model  


def early_bird_lottery_ticket(
    prune_epoch,
    sparsity,
    stability_threshold,
    model,
    num_epochs,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    experiment,
    checkpoint_path
):
    fifo_queue = deque(maxlen=5)
    mask_prev = None

    model.to(device)

    for e in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {e + 1}, Loss: {avg_loss:.4f}")

        if e == prune_epoch:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=sparsity,
                        n=1,
                        dim=0
                    )

            mask = {
                name: module.weight_mask.clone()
                for name, module in model.named_modules() if hasattr(module, "weight_mask")
            }

            if mask_prev is not None:
                distance = mask_distance(mask, mask_prev)
                fifo_queue.append(distance)

                if max(fifo_queue) < stability_threshold:
                    print(f"Early Bird Lottery Ticket found at epoch {e + 1}!")
                    torch.save(model.state_dict(), f"{experiment}_early_bird_model.pth")
                    torch.save(mask, f"{experiment}_early_bird_mask.pth")
                    return model, mask
            
            mask_prev = mask
            torch.save(mask, f"{checkpoint_path}/{experiment}_iter_{e+1}_mask.pth")
        
        scheduler.step()

        checkpoint = {
            'epoch': e + 1,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'sched': scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_path}/epoch_{e+1}.pth")

    return model, None

def mask_distance(mask1, mask2):
    total_change = 0
    for name in mask1:
        total_change += (mask1[name] != mask2[name]).sum().item()
    return total_change

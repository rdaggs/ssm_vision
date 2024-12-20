import torch
import time

def train_model(model, train_loader, test_loader, criterion, optimizer,scheduler, num_epochs, device, experiment, prune_gradients, checkpoint_path=None):
    
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

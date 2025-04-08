import os
import torch

# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, fold, train_losses, val_losses, 
                   train_accs, val_accs, best_val_acc, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'fold': fold,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

# Load checkpoint
def load_checkpoint(model, optimizer, scheduler, filepath):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return model, optimizer, scheduler, 0, 0, [], [], [], [], 0

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return (model, optimizer, scheduler, checkpoint['epoch'], checkpoint['fold'],
            checkpoint['train_losses'], checkpoint['val_losses'],
            checkpoint['train_accs'], checkpoint['val_accs'], checkpoint['best_val_acc'])

import os
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint

# Training loop with KFold and resume
def train_with_kfold_resumable(model_class, combined_dataset, num_folds=5, batch_size=32,
                               num_epochs=100, patience=15, learning_rate=5e-5, weight_decay=1e-4,
                               checkpoint_dir='checkpoints', resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    start_fold = 0
    fold_info_path = os.path.join(checkpoint_dir, 'fold_info.pt')
    if resume and os.path.exists(fold_info_path):
        fold_info = torch.load(fold_info_path)
        fold_accuracies = fold_info.get('fold_accuracies', [])
        start_fold = fold_info.get('next_fold', 0)
        print(f"Resuming from fold {start_fold}")

    fold_splits = list(kfold.split(combined_dataset))

    for fold, (train_ids, val_ids) in enumerate(fold_splits):
        if fold < start_fold:
            print(f"Skipping fold {fold+1} (already completed)")
            continue

        print(f"FOLD {fold+1}/{num_folds}")
        print('-' * 40)

        train_loader = DataLoader(combined_dataset, batch_size=batch_size,
                                  sampler=SubsetRandomSampler(train_ids))
        val_loader = DataLoader(combined_dataset, batch_size=batch_size,
                                sampler=SubsetRandomSampler(val_ids))

        model = model_class(num_classes=8).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay, amsgrad=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=learning_rate,
                                                        steps_per_epoch=len(train_loader),
                                                        epochs=num_epochs,
                                                        pct_start=0.2)

        checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold+1}_checkpoint.pt')
        best_model_path = os.path.join(checkpoint_dir, f'best_model_fold_{fold+1}.pth')

        start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc = 0, [], [], [], [], 0.0
        if resume and os.path.exists(checkpoint_path):
            model, optimizer, scheduler, start_epoch, _, train_losses, val_losses, train_accs, val_accs, best_val_acc = \
                load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            if start_epoch > 0:
                for _ in range(start_epoch * len(train_loader)):
                    scheduler.step()

        epochs_without_improvement, early_stop = 0, False
        try:
            for epoch in range(start_epoch, num_epochs):
                if early_stop:
                    break

                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                train_losses.append(running_loss / len(train_loader))
                train_accs.append(correct / total)

                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)

                val_losses.append(val_loss / len(val_loader))
                val_acc = val_correct / val_total
                val_accs.append(val_acc)

                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f}, "
                      f"Train Acc: {train_accs[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

                save_checkpoint(model, optimizer, scheduler, epoch+1, fold,
                                train_losses, val_losses, train_accs, val_accs, best_val_acc,
                                checkpoint_path)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Model improved! Saved best model for fold {fold+1}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        early_stop = True

            fold_accuracies.append(best_val_acc)
            torch.save({'fold_accuracies': fold_accuracies, 'next_fold': fold + 1}, fold_info_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            save_checkpoint(model, optimizer, scheduler, epoch, fold,
                            train_losses, val_losses, train_accs, val_accs, best_val_acc,
                            checkpoint_path)
            torch.save({'fold_accuracies': fold_accuracies, 'next_fold': fold}, fold_info_path)
            return fold_accuracies

    print(f"\nK-Fold Cross Validation Results:\nMean val acc: {np.mean(fold_accuracies):.4f} | "
          f"Std dev: {np.std(fold_accuracies):.4f}")
    return fold_accuracies


def final_evaluation(model_class, fold_accuracies, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_fold = np.argmax(fold_accuracies) + 1
    print(f"Best fold: {best_fold} with acc: {fold_accuracies[best_fold-1]:.4f}")
    model = model_class(num_classes=8).to(device)
    model.load_state_dict(torch.load(f'best_model_fold_{best_fold}.pth'))

    model.eval()
    test_correct, test_total = 0, 0
    class_correct = [0] * 8
    class_total = [0] * 8

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            for i in range(len(labels)):
                class_correct[labels[i]] += (preds[i] == labels[i]).item()
                class_total[labels[i]] += 1

    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    for i in range(8):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Class {i} accuracy: {class_acc:.4f}")
    torch.save(model.state_dict(), 'final_model.pth')
    print("Final model saved as 'final_model.pth'")
    return test_acc, model

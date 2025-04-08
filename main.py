import torch
from model import HybridCCT
from train_eval import train_with_kfold_resumable, final_evaluation
from data_loader import get_dataloaders

# Get datasets
data_dir = "/home/srmist8/PROJECT_MAJOR/smoted_images_split"
combined_dataset, test_loader = get_dataloaders(data_dir, batch_size=32)


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    NUM_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    PATIENCE = 15
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4

    print("Starting/Resuming K-fold cross-validation...")

    fold_accuracies = train_with_kfold_resumable(
        model_class=HybridCCT,
        combined_dataset=combined_dataset,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir='model_checkpoints',
        resume=True
    )

    final_acc, best_model = final_evaluation(
        model_class=HybridCCT,
        fold_accuracies=fold_accuracies,
        test_loader=test_loader
    )

    print(f"Cross-validation complete! Final test accuracy: {final_acc:.4f}")

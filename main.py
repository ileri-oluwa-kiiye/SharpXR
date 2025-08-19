import torch
from data.dataset import XrayDataset
from models.dual_decoder import DualDecoderHybrid
from utils.data_loading import create_data_loaders
from training.trainer import train_model
from evaluation.evaluator import evaluate_model
from config.configs import (
    DATA_ROOT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, 
    BATCH_SIZE, MODEL_IN_CHANNELS, MODEL_OUT_CHANNELS, MODEL_FEATURES,
    EPOCHS, LEARNING_RATE, PATIENCE, USE_PSNR_SSIM,
    BEST_MODEL_PATH, TRAINING_SESSION_PATH, SAVE_ALL_AS_PKL
)


def main():
    """Main function to set up data, model, training, and evaluation"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the dataset
    print(f"Loading dataset from: {DATA_ROOT_DIR}")
    dataset = XrayDataset(DATA_ROOT_DIR)
    print(f"Dataset loaded with {len(dataset)} images")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        batch_size=BATCH_SIZE
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    model = DualDecoderHybrid(
        in_channels=MODEL_IN_CHANNELS,
        out_channels=MODEL_OUT_CHANNELS,
        features=MODEL_FEATURES
    ).to(device)
    
    print(f"Model initialized: {model.__class__.__name__}")
    
    # Train model
    print("\nStarting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        use_psnr_ssim=USE_PSNR_SSIM,
        patience=PATIENCE,
        save_path=BEST_MODEL_PATH,
        save_all_as_pkl=SAVE_ALL_AS_PKL,
        pkl_path=TRAINING_SESSION_PATH,
        device=device
    )
    
    # Evaluate model
    print("\nStarting evaluation...")
    evaluation_results = evaluate_model(trained_model, test_loader, device=device)
    
    return trained_model, evaluation_results


if __name__ == "__main__":
    model, results = main()
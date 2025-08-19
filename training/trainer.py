import torch
import torch.optim as optim
import pickle
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.metrics import calculate_snr, rmse_loss


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=1e-3,
    use_psnr_ssim=True,
    patience=30,
    save_path="best_model.pt",
    save_all_as_pkl=True,
    pkl_path="training_session.pkl",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the denoising model
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        use_psnr_ssim: Whether to compute PSNR/SSIM metrics
        patience: Early stopping patience
        save_path: Path to save best model
        save_all_as_pkl: Whether to save training data as pickle
        pkl_path: Path to save pickle file
        device: Device to use for training
    
    Returns:
        Trained model
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses, val_losses = [], []
    val_psnr_scores, val_ssim_scores, val_snr_scores = [], [], []

    best_val_score = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for noisy_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = rmse_loss(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} | Train RMSE: {avg_train_loss:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0.0
            total_psnr, total_ssim, total_snr = 0.0, 0.0, 0.0

            with torch.no_grad():
                for noisy_imgs, clean_imgs in val_loader:
                    noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                    outputs = model(noisy_imgs)
                    loss = rmse_loss(outputs, clean_imgs)
                    val_loss += loss.item()

                    if use_psnr_ssim:
                        preds_np = outputs.squeeze(1).cpu().numpy()
                        targets_np = clean_imgs.squeeze(1).cpu().numpy()
                        for i in range(preds_np.shape[0]):
                            total_psnr += psnr(targets_np[i], preds_np[i], data_range=1.0)
                            total_ssim += ssim(targets_np[i], preds_np[i], data_range=1.0)
                            total_snr += calculate_snr(preds_np[i], targets_np[i])

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if use_psnr_ssim:
                num_val_imgs = len(val_loader.dataset)
                avg_psnr = total_psnr / num_val_imgs
                avg_ssim = total_ssim / num_val_imgs
                avg_snr = total_snr / num_val_imgs
                val_psnr_scores.append(avg_psnr)
                val_ssim_scores.append(avg_ssim)
                val_snr_scores.append(avg_snr)
                print(f"Val RMSE: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, SNR: {avg_snr:.2f}")
            else:
                print(f"Val RMSE: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_score:
                best_val_score = avg_val_loss
                best_model_state = model.state_dict()
                torch.save(best_model_state, save_path)
                print("Best model saved.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement. Patience counter: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    if save_all_as_pkl:
        training_data = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_psnr": val_psnr_scores,
            "val_ssim": val_ssim_scores,
            "val_snr": val_snr_scores,
            "best_model_path": save_path,
            "best_val_rmse": best_val_score,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(training_data, f)
        print(f"Session saved as {pkl_path}")

    return model
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.metrics import calculate_snr


def evaluate_model(model, dataloader, compute_metrics=True, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the denoising model
    
    Args:
        model: The trained neural network model
        dataloader: Test data loader
        compute_metrics: Whether to compute image quality metrics
        device: Device to use for evaluation
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    model.to(device)
    rmse_loss = torch.nn.MSELoss(reduction='mean')
    count = 0

    # Accumulators for noisy image vs clean
    total_psnr_noisy = 0.0
    total_ssim_noisy = 0.0
    total_snr_noisy = 0.0

    # Accumulators for denoised image vs clean
    total_rmse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_snr = 0.0

    with torch.no_grad():
        for noisy_imgs, clean_imgs in tqdm(dataloader, desc="Evaluating"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            outputs = model(noisy_imgs)

            preds_np = outputs.squeeze(1).cpu().numpy()
            targets_np = clean_imgs.squeeze(1).cpu().numpy()
            noisy_np = noisy_imgs.squeeze(1).cpu().numpy()

            for i in range(preds_np.shape[0]):
                # Denoised image metrics
                mse = ((preds_np[i] - targets_np[i]) ** 2).mean()
                rmse = np.sqrt(mse)
                total_rmse += rmse

                if compute_metrics:
                    total_psnr += psnr(targets_np[i], preds_np[i], data_range=1.0)
                    total_ssim += ssim(targets_np[i], preds_np[i], data_range=1.0)
                    total_snr += calculate_snr(preds_np[i], targets_np[i])

                    # Noisy image metrics
                    total_psnr_noisy += psnr(targets_np[i], noisy_np[i], data_range=1.0)
                    total_ssim_noisy += ssim(targets_np[i], noisy_np[i], data_range=1.0)
                    total_snr_noisy += calculate_snr(noisy_np[i], targets_np[i])

            count += preds_np.shape[0]

    # Calculate averages
    avg_rmse = total_rmse / count
    avg_psnr = total_psnr / count if compute_metrics else None
    avg_ssim = total_ssim / count if compute_metrics else None
    avg_snr = total_snr / count if compute_metrics else None
    avg_psnr_noisy = total_psnr_noisy / count if compute_metrics else None
    avg_ssim_noisy = total_ssim_noisy / count if compute_metrics else None
    avg_snr_noisy = total_snr_noisy / count if compute_metrics else None

    # Print results
    print("\nFinal Evaluation Results:")
    print(f"RMSE (denoised vs. clean): {avg_rmse:.4f}")
    
    if compute_metrics:
        print("\nNoisy image vs. Clean:")
        print(f"PSNR: {avg_psnr_noisy:.2f}, SSIM: {avg_ssim_noisy:.4f}, SNR: {avg_snr_noisy:.2f}")
        print("\nDenoised image vs. Clean:")
        print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, SNR: {avg_snr:.2f}")

    return {
        "rmse": avg_rmse,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "snr": avg_snr,
        "psnr_noisy": avg_psnr_noisy,
        "ssim_noisy": avg_ssim_noisy,
        "snr_noisy": avg_snr_noisy
    }
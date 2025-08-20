# SharpXR: Structure-Aware Denoising for Pediatric Chest X-Rays

SharpXR is a **dual-decoder U-Net** for denoising **low-dose pediatric chest X-rays** while preserving diagnostically relevant structure. This repo contains the implementation, training and evaluation scripts used in our MIRASOL, MICCAI 2025 work.

> *SharpXR: Structure-Aware Denoising for Pediatric Chest X-Rays*  
> Code: https://github.com/ileri-oluwa-kiiye/SharpXR

---

## Key Features
- **Dual-decoder U-Net**: one decoder for noise suppression, one Laplacian-enhanced decoder for edge preservation.  
- **Learnable fusion**: pixel-wise attention weights balance smoothness vs. structure.  
- **Realistic noise simulation**: hybrid **Poisson–Gaussian** model for low-dose conditions.  
- **Comprehensive benchmarks**: BM3D, DnCNN, REDCNN, ResUNet++, Attention U-Net, Sharp U-Net, Hformer.  
- **Downstream utility**: boosts pneumonia classification accuracy (88.8% → 92.5%).

---

## Repository Structure
    ├── checkpoints/        # Pretrained weights / saved states
    ├── config/             # YAML/JSON configs (training, eval, paths, noise)
    ├── data/               # Datasets & loaders
    ├── evaluation/         # Metrics (RMSE, PSNR, SSIM, SNR) + scripts
    ├── models/             # SharpXR modules and baselines
    ├── training/           # Trainer, loops, logging
    ├── utils/              # Preprocessing, augmentation, misc helpers
    ├── main.py             # Entry point (train/eval)
    └── README.md

---

## Installation
    git clone https://github.com/ileri-oluwa-kiiye/SharpXR.git
    cd SharpXR
    pip install -r requirements.txt

**Requires**: Python 3.8+, PyTorch, torchvision, numpy, scikit-image, tqdm.

---

## Dataset
We use the **Pediatric Pneumonia Chest X-ray** dataset (Kermany et al., 2018):  
[https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)

Preprocessing:
- Resize to **256×256**, normalize to [0,1].
- Split: 75% train / 10% val / 15% test (stratified).
- Augment: horizontal flip (p=0.5), ±10% brightness/contrast, ±15° rotation.

Configure dataset paths in `config/dataset.yaml`.

**Noise model (applied to training inputs)**  
Poisson–Gaussian:
\[
\tilde{X} = \frac{1}{\eta}\,\mathrm{Poisson}(\eta X) + \mathcal{N}(0,\sigma^2),\;
\eta \in [50,300],\; \sigma \in [5,30]
\]

---

## Usage

### Train
    python main.py --config config/train.yaml

### Evaluate
    python main.py --config config/eval.yaml --checkpoint checkpoints/sharpxr_best.pth

### Metrics reported
RMSE, PSNR, SSIM, SNR (logged to `evaluation/`).

---

## Results

**Overall benchmark on pediatric chest X-rays (lower RMSE is better; higher PSNR/SSIM/SNR is better).**

| Model        | RMSE ↓  | PSNR ↑  | SSIM ↑  | SNR ↑  |
|--------------|---------|---------|---------|--------|
| BM3D         | 0.0346  | 29.45   | 0.7003  | 22.85  |
| DnCNN        | 0.0203  | 33.97   | 0.8945  | 28.29  |
| REDCNN       | 0.0181  | 34.95   | 0.9140  | 29.20  |
| Sharp U-Net  | 0.0172  | 35.40   | 0.9261  | 29.72  |
| **SharpXR**  | **0.0170** | **35.52** | **0.9263** | **29.84** |

**Downstream classification**  
Using SharpXR-denoised images improves pneumonia classification accuracy from **88.8% → 92.5%** (clean: 93.7%).

**Qualitative examples**  
See `evaluation/qualitative_results/` for ROI crops highlighting rib and lung boundary preservation.

---

## Reproducibility Notes
- Optimizer: Adam (lr=1e-4), batch size=4, epochs=50.  
- Fixed random seeds in training scripts for determinism (subject to CUDA/cuDNN).

---

## Citation
If this repository helps your research, please cite:

    @inproceedings{abolade2025sharpxr,
      title   = {SharpXR: Structure-Aware Denoising for Pediatric Chest X-Rays},
      author  = {Abolade, Ilerioluwakiiye and Idoko, Emmanuel and Odelola, Solomon and Omoigui, Promise and Adebanwo, Adetola and Iorumbur, Aondana and Anazodo, Udunna and Crimi, Alessandro and Confidence, Raymond},
      booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
      year    = {2025}
    }

---

## Acknowledgements
We thank **ML Collective** for compute and weekly feedback, and **AFRICAI** for mentorship.  
Supported in part by the Italian Ministry of University and Research (MUR), project **PE0000013 – FAIR**.

---


## Links
- Code: https://github.com/ileri-oluwa-kiiye/SharpXR
- Paper: [https://www.arxiv.org/abs/2508.08518](https://www.arxiv.org/abs/2508.08518)

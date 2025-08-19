DATA_ROOT_DIR = "/kaggle/working/datasets/"
IMAGE_SIZE = (256, 256)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training settings
BATCH_SIZE = 4
SHUFFLE_TRAIN = True

# Model settings
MODEL_IN_CHANNELS = 1
MODEL_OUT_CHANNELS = 1
MODEL_FEATURES = [64, 128, 256, 512]

# Noise parameters
POISSON_SCALE_RANGE = (50, 300)
GAUSSIAN_STD_RANGE = (5, 25)

# Augmentation parameters
FLIP_PROB = 0.5
ROTATION_RANGE = (-15, 15)
BRIGHTNESS_RANGE = (0.9, 1.1)
CONTRAST_RANGE = (0.9, 1.1)

# Training hyperparameters
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5
USE_PSNR_SSIM = True

# Save paths
BEST_MODEL_PATH = "best_denoiser.pt"
TRAINING_SESSION_PATH = "denoising_training.pkl"
SAVE_ALL_AS_PKL = True
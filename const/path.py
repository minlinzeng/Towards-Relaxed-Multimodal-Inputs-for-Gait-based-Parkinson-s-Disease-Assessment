import os

# Redirect to valid local paths
PROJECT_ROOT = os.getcwd()  # Use current working directory as fallback
NDRIVE_PROJECT_ROOT = os.path.join(PROJECT_ROOT, 'motion_evaluator')

# Stores pre-trained model checkpoints for the motion encoders used in the project.
PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = os.path.join(NDRIVE_PROJECT_ROOT, 'Pretrained_checkpoints')
# Directory for saving project outputs, such as logs, results, or processed data.
OUT_PATH = os.path.join(PROJECT_ROOT, 'log/motion_encoder/out')

# Kinect
# Directory for preprocessed data.
PREPROCESSED_DATA_ROOT_PATH = os.path.join(PROJECT_ROOT, 'PD_3D_motion-capture_data/C3Dfiles_processed')

# PD
PD_PATH_POSES = os.path.join(PROJECT_ROOT, 'PD_3D_motion-capture_data/C3Dfiles_processed_new/') # Directory containing preprocessed Parkinson’s Disease dataset pose files in .npy format.
PD_PATH_SENSORS = os.path.join(PROJECT_ROOT, 'PD_3D_motion-capture_data/GRF_processed/') # Directory containing preprocessed Parkinson’s Disease dataset sensor files in .npy format.
PD_PATH_LABELS = os.path.join(PROJECT_ROOT, 'PD_3D_motion-capture_data/PDGinfo.xlsx') # Path to a CSV file containing metadata and labels for the Parkinson’s Disease dataset.

# Directory for saving checkpoints during training or fine-tuning of models.
CHECKPOINT_ROOT_PATH = os.path.join(PROJECT_ROOT, 'log/motion_encoder/out/motionbert/finetune_6_pd,json/1/models')


os.makedirs(PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_ROOT_PATH, exist_ok=True)
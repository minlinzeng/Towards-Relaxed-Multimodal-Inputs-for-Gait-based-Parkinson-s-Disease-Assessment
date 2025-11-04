import os
import numpy as np
import torch
from data.public_pd_datareader import PDReader
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ProcessedDataset(Dataset):
    def __init__(self, pose_dir, sensor_dir, labels_path, sensor_encoder, transform=None):
        """
        pose_dir: Directory with processed skeleton npy files.
        sensor_dir: Directory with processed sensor npy files.
        labels_path: CSV file mapping subject IDs to labels.
        sensor_encoder: A torch.nn.Module that converts sensor data from (101, T, 3) 
                        to a fixed shape (101, fixed_dim).
        transform: Optional transform to apply on each sample.
        """
        self.pose_files = sorted([os.path.join(pose_dir, f) for f in os.listdir(pose_dir) if f.endswith('.npy')])
        self.sensor_files = sorted([os.path.join(sensor_dir, f) for f in os.listdir(sensor_dir) if f.endswith('.npy')])
        
        # Load labels from CSV: assume columns: "Subject", "Label"
        df = pd.read_csv(labels_path)
        self.labels_dict = {row['Subject']: row['Label'] for _, row in df.iterrows()}
        
        # Align files based on subject ID (assumed to be first token of filename)
        self.data = []  # Each element is (pose_file, sensor_file, subject_id)
        for pose_file in self.pose_files:
            subj = os.path.basename(pose_file).split('_')[0]
            sensor_file = next((sf for sf in self.sensor_files if os.path.basename(sf).startswith(subj)), None)
            if sensor_file:
                self.data.append((pose_file, sensor_file, subj))
            else:
                print(f"[WARN] No sensor file found for subject {subj}")
        
        self.sensor_encoder = sensor_encoder  # Must be a fixed encoder module
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pose_file, sensor_file, subj = self.data[idx]
        
        # Load skeleton data, expected fixed shape, e.g., (101, n_joints, 3)
        pose = np.load(pose_file)
        # Load sensor data, variable shape: (101, T, 3)
        sensor = np.load(sensor_file)
        
        # Get label for this subject
        label = self.labels_dict.get(subj, 0)
        
        # Convert to torch tensors.
        pose_tensor = torch.tensor(pose, dtype=torch.float32)
        sensor_tensor = torch.tensor(sensor, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply sensor encoder to get fixed-size sensor representation.
        # The sensor_encoder converts sensor_tensor from (101, T, 3) to, say, (101, fixed_dim)
        sensor_fixed = self.sensor_encoder(sensor_tensor)
        
        sample = {
            'skeleton': pose_tensor,
            'sensor': sensor_fixed,
            'label': label_tensor,
            'subject': subj
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Example usage:
if __name__ == "__main__":
    # Define paths.
    pose_dir = "./PD_3D_motion-capture_data./C3Dfiles_processed_new"
    sensor_dir = "./PD_3D_motion-capture_data./GRF_processed"
    labels_path = "./PD_3D_motion-capture_data./PDGinfo.csv"
    
    # Import your sensor encoder; here CombinedSensorEncoder processes left/right if needed,
    # but if you already combined sensor data into one file, use a SensorTemporalEncoder variant.
    from feature_encoder import CombinedSensorEncoder
    # For example, here we assume sensor_encoder converts (101, T, 3) to (101, 32)
    sensor_encoder = CombinedSensorEncoder(input_channels=3, hidden_dim=16, output_dim=32, kernel_size=3)
    sensor_encoder.eval()  # Optionally set to eval mode if not training it jointly.
    
    dataset = ProcessedDataset(pose_dir, sensor_dir, labels_path, sensor_encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Check one batch.
    for batch in dataloader:
        print("Skeleton batch shape:", batch['skeleton'].shape)  # e.g., (B, 101, n_joints, 3)
        print("Sensor batch shape:", batch['sensor'].shape)      # e.g., (B, 101, fixed_dim)
        print("Labels:", batch['label'])
        break

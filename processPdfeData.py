import os
import json
import numpy as np
from data.pd.preprocess_pd import visualize_sequence
from data.dataloaders import ProcessedDataset
from sklearn.model_selection import StratifiedKFold
import pandas as pd

class pdfeReader():
    skeleton = [
        [0, 1], [1, 2], [2, 3],
        [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13],
        [8, 14], [14, 15], [15, 16]
    ]
    
    def __init__(self, pose_path, sensor_path, label_path, lifted_path, pose_seg=36, sensor_seg=36, downsample_factor=3):
        """_summary_
        Returns:
            pose_dict (dict): {'SUB01_1_1':[], 'SUB02_1_cropped_1':[]} shape: (100, 7, 3)
            sensor_dict (dict):  {'SUB01_1_1':[], 'SUB02_1_cropped_1':[]} shape: (142, 6)
            labels_dict (dict): {'SUB01': [1,1]}
            sensor_length (int): The length of the longest sensor segment.
        """
        self.pose_path      = pose_path
        self.sensor_path    = sensor_path
        self.label_path     = label_path
        self.lifted_path    = lifted_path
        self._pose_cache    = {}
        self._sensor_cache  = {}
        self.sensor_length  = 0
        self.pose_segs      = pose_seg
        self.sensor_segs    = sensor_seg
        self.downsample_factor = downsample_factor
        self.loaded_data    = {}
        
        # self.label_list   = pd.read_excel(self.labels_path, engine='openpyxl')
        self.sensor_dict, self.sensor_length = self.load_sensor_data()
        self.labels_dict    = self.load_subject_labels()
        self.pose_dict      = self.load_pose_data()
        self.pose_preprocess()
        self.label_preprocess()
        self.sensor_preprocess()
        print(f"self.sensor_length is :{self.sensor_length}")

    def pose_preprocess(self):
        """
            manually discard pose data segments that doesn't have correct skeleton projection (confused with others)
            : 21-1 [10:], 21-3 [40:], 24-1 [12:32],[1:05:]
        """
        if 'SUB21_1_1' in self.pose_dict:
            self.pose_dict.pop('SUB21_1_1')
            for i in range(14):
                self.pose_dict.pop(f'SUB21_3_{i+1}')
            for i in range(4):
                self.pose_dict.pop(f'SUB24_1_{i+1}')
            for i in range(10, 20):
                self.pose_dict.pop(f'SUB24_1_{i+1}')
        
    def sensor_preprocess(self):
        if 'SUB19_1_1' in self.sensor_dict:
            self.sensor_dict.pop('SUB19_1_1')
        
    def label_preprocess(self):
        if 'SUB21' in self.labels_dict:
            self.labels_dict['SUB19'] = [2]

    def load_pose_data(self):
        pose_dict = {}
        lifted_video_names = [f.replace('.mp4','') for f in os.listdir(self.lifted_path) if f.endswith('.mp4')]
        for file in os.listdir(self.pose_path):
            if not file.endswith('.json'):
                continue
            # print(f"[INFO] Processing {file}")
            video_name = file.replace('_3d_predictions.json', '')  # e.g., PDFE01_1
            if video_name not in lifted_video_names:
                continue
            
            video_name = video_name.replace('PDFE', 'SUB')  # e.g., SUB01_1
            
            with open(os.path.join(self.pose_path, file), 'r') as f:
                data = json.load(f)

            # Extract keypoints for each frame
            frames = []
            for frame_pred in data:
                instances = frame_pred.get('predictions') or []
                if not instances:
                    continue

                instance = instances[0]  # First detected person
                keypoints = instance[0]['keypoints'][0:7]
                frames.append(keypoints)

            sequence = np.array(frames)  # shape: (num_frames, 7, 3)
            total_frames = sequence.shape[0]
            segment_len = total_frames // self.pose_segs
            if segment_len == 0:
                print(f"[WARN] Skipping {video_name} — not enough frames to split into {self.pose_segs} segments.")
                continue

            for i in range(self.pose_segs):
                start = i * segment_len
                end = (i + 1) * segment_len if i < self.pose_segs - 1 else total_frames
                segment = sequence[start:end]
                if segment.shape[0] < 1:
                    continue
                video_name = video_name.replace('_cropped','')
                segment_name = f"{video_name}_{i + 1}"
                pose_dict[segment_name] = segment
        return pose_dict

    def load_sensor_data(self):
        raw_sensor_dict = {}
        sensor_dict = {}
        sensor_length = 0

        for fname in os.listdir(self.sensor_path):
            if not fname.endswith(".txt") or "standing" in fname.lower():
                continue  # Skip non-IMU trials
            # print(f"[INFO] Processing {fname}")
            name_part = fname.replace(".txt", "")  # e.g., "SUB01_1"
            file_path = os.path.join(self.sensor_path, fname)

            try:
                df = pd.read_csv(file_path, sep=r'\s{2,}|\t', engine='python')
                raw_signal = df.iloc[:, 2:8].to_numpy()  # shape (T, 6)
                # Downsampling here
                raw_signal = raw_signal[::self.downsample_factor, :]
                raw_sensor_dict[name_part] = raw_signal
            except Exception as e:
                print(f"[ERROR] Failed to read {fname}: {e}")
                continue

        # 🔁 Segment each sensor trial into N parts
        for name, signal in raw_sensor_dict.items():
            total_samples = signal.shape[0]
            segment_len = total_samples // self.sensor_segs
            if segment_len == 0:
                print(f"[WARN] Skipping {name} — not enough samples for {self.sensor_segs} segments.")
                continue

            for i in range(self.sensor_segs):
                start = i * segment_len
                end = (i + 1) * segment_len if i < self.sensor_segs - 1 else total_samples
                segment = signal[start:end]
                if segment.shape[0] < 1: continue
                segment_name = f"{name}_{i + 1}"
                sensor_dict[segment_name] = segment
                if segment.shape[0] > sensor_length:
                    sensor_length = segment.shape[0]

        print(f"[INFO] Sensor segmentation complete: {len(sensor_dict)} segments generated.")
        return sensor_dict, sensor_length

    def load_subject_labels(self):
        """
        Load subject labels from the Excel file.

        Returns:
            labels_dict (dict): A dictionary mapping subject IDs to their labels.
            {'SUB01': [0, 1], 'SUB02': [1], ...}
        """
        label_df = pd.read_excel(self.label_path)
        label_df.columns = [str(col).strip() for col in label_df.columns]
        hy_columns = [col for col in label_df.columns if "H&Y" in col]

        subject_labels = {}
        for idx, row in label_df.iterrows():
            if idx == 0:
                continue  # skip header row if needed
            subject_id = f"SUB{idx:02d}"
            labels = []
            for col in hy_columns:
                try:
                    if pd.notna(row[col]):
                        labels.append(int(row[col]) - 2)  # normalize label (e.g., 2→0, 3→1)
                except ValueError:
                    continue
            if labels:  # only add if there's at least one label
                subject_labels[subject_id] = labels
        return subject_labels

    def visualize_pose():
        # ---------------- Loop through poses -----------------
        for json_file in json_files:
            if json_file.endswith('_3d_predictions.json'):
                base_name = json_file.replace('_3d_predictions.json', '')
                if base_name in video_names:
                    print(f"Processing {json_file} for {base_name}.mp4")
                    # with open(os.path.join(pose_path, json_file), 'r') as f:
                    with open(os.path.join(pose_path, 'PDFE03_2_3d_predictions.json'), 'r') as f:
                        pose = json.load(f)
                    keypoints_seq = []

                    for frame_pred in pose:
                        instances = frame_pred.get('predictions') or []
                        if not instances:
                            continue

                        instance = instances[0]  # First detected person
                        keypoints = instance[0]['keypoints']
                        keypoints_seq.append(keypoints)

                    keypoints_seq = np.array(keypoints_seq[:1000])  # (num_frames, 17, 3)

                    visualize_sequence(keypoints_seq, name=save_name)
                    break  # Process only one video-pose pair for now

def main():
    reader = pdfeReader(
        pose_path='/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/predictions/',
        lifted_path='/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/lifted/',
        sensor_path='/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/IMU/',
        label_path='/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/PDFEinfo.xlsx'
    )
    print(f"sensor length is :{reader.sensor_length}")

if __name__ == "__main__":
    main()        
    # video_path = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/lifted/'
    # pose_path = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/predictions/'
    # sensor_path = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/IMU/'
    # label_path = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/PDFEinfo.xlsx'
    # video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
    # json_files = [f for f in os.listdir(pose_path) if f.endswith('.json')]
    # video_names = set(os.path.splitext(f)[0] for f in video_files)
    # save_name = './skeleton_motion'
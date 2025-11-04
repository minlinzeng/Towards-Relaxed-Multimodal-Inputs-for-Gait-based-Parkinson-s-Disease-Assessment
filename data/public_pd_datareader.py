import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import *

class PDReader():
    """
    Reads processed .npy skeleton data from preproceed_pd.py under pd folder, assigns labels, extracts metadata, and structures data for further use (dataloaders.py).
    """

    ON_LABEL_COLUMN = 'ON - UPDRS-III - walking'
    OFF_LABEL_COLUMN = 'OFF - UPDRS-III - walking'
    DELIMITER = ';'

    def __init__(self, joints_path, sensor_path, labels_path):
        self.joints_path = joints_path
        self.sensor_path = sensor_path
        self.labels_path = labels_path
        self._pose_cache = {}
        self._sensor_cache = {}
        
        self.label_list = pd.read_excel(self.labels_path, engine='openpyxl')
        self.sensor_dict, self.sensor_label_dict = self.read_sensor_data(sensor_path)
        self.pose_dict, self.pose_label_dict, self.video_names, self.metadata_dict = self.read_keypoints_and_labels() 

    def read_sensor_data(self, sensor_path):
        """
        Reads sensor data npy files from the sensor_path folder. (SUB01_on_left.npy).
        Returns a dictionary mapping subject IDs to sensor data arrays.
        """
        sensor_dict = {}
        sensor_label_dict = {}
        for file in os.listdir(sensor_path):
            if file.endswith('.npy'):
                # Expecting filenames like "SUB01_on_left.npy" or similar.
                subj_id = file.split('_')[0]  # "SUB01"
                on_or_off = file.split('_')[1]
                left_or_right = file.split('_')[2].split('.')[0]  # "left or right"
                key = f"{subj_id}_{on_or_off}_{left_or_right}"
                
                if key in self._sensor_cache:
                    sensor_data = self._sensor_cache[key]
                else:
                    full_path = os.path.join(sensor_path, file)
                    sensor_data = np.load(full_path, allow_pickle=True)
                    self._sensor_cache[key] = sensor_data  # Store in cache
                
                if sensor_data.shape[1] != 0:
                    sensor_dict[key] = sensor_data
                    sensor_label_dict[key] = self.read_label(file)
        
        return sensor_dict, sensor_label_dict
        # return sensor_dict, empty_sensor

    def read_sequence(self, path_file):
        """
        Reads skeletons from npy files and convert them to meters.
        """
        if path_file in self._pose_cache:
            return self._pose_cache[path_file]
        
        if os.path.exists(path_file):
            body = np.load(path_file)
            body = body/1000 #convert mm to m
            self._pose_cache[path_file] = body
        else:
            body = None
        return body

    def read_label(self, file_name):
        """
        Reads the label (0,1,2) from the labels file.
        """
        subject_id, on_or_off = file_name.split("_")[:2]
        df = self.label_list
        df = df[['ID', self.ON_LABEL_COLUMN, self.OFF_LABEL_COLUMN]]
        subject_rows = df[df['ID'] == subject_id]
        if on_or_off == "on":
            label = subject_rows[self.ON_LABEL_COLUMN].values[0]
        else:
            label = subject_rows[self.OFF_LABEL_COLUMN].values[0]
        return int(label)
    
    def read_metadata(self, file_name):
        #If you change this function make sure to adjust the METADATA_MAP in the dataloaders.py accordingly
        subject_id = file_name.split("_")[0]
        df = pd.read_excel(self.labels_path, engine='openpyxl')
        df = df[['ID', 'Gender', 'Age', 'Height (cm)', 'Weight (kg)', 'BMI (kg/m2)']]
        df.rename(columns={
            "Gender": "gender",
            "Age": "age",
            "Height (cm)": "height",
            "Weight (kg)": "weight",
            "BMI (kg/m2)": "bmi"}, inplace=True)
        df.loc[:, 'gender'] = df['gender'].map({'M': 0, 'F': 1})
        
        # Using Min-Max normalization
        df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        df['height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
        df['weight'] = (df['weight'] - df['weight'].min()) / (df['weight'].max() - df['weight'].min())
        df['bmi'] = (df['bmi'] - df['bmi'].min()) / (df['bmi'].max() - df['bmi'].min())

        subject_rows = df[df['ID'] == subject_id]
        return subject_rows.values[:, 1:] 
    
    def read_keypoints_and_labels(self):
        """
        Read npy files in given directory into arrays of pose keypoints.
        return: dictionary with <key=video name, value=keypoints>
        """
        pose_dict = {}
        pose_label_dict = {}
        metadata_dict = {}
        video_names_list = [] # 'SUB23_on_walk_10_0'

        print('[INFO - PublicPDReader] Reading body keypoints from npy')
        print(self.joints_path)
        
        for file_name in tqdm(os.listdir(self.joints_path)):
            path_file = os.path.join(self.joints_path, file_name)
            joints = self.read_sequence(path_file)
            label = self.read_label(file_name)
            metadata = self.read_metadata(file_name)
            if joints is None:
                print(f"[WARN - PublicPDReader] Numpy file {file_name} does not exist")
                continue
            file_name = file_name.split(".")[0]
            pose_dict[file_name] = joints
            pose_label_dict[file_name.split("_")[0]+"_"+file_name.split("_")[1]] = label
            metadata_dict[file_name] = metadata
            video_names_list.append(file_name)

        return pose_dict, pose_label_dict, video_names_list, metadata_dict

    @staticmethod
    def select_unique_entries(a_list):
        return sorted(list(set(a_list)))

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        print("getitem in PDReader is empty. Try accessing elements directly.")
        pass


# raw_data = PDReader('/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/C3Dfiles_processed_new', '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/GRF_processed', '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/PDGinfo.xlsx') 

# print(raw_data.video_names[0], raw_data.participant_ID[0], raw_data.pose_dict.popitem(), raw_data.sensor_dict.popitem(), raw_data.labels_dict.popitem(), raw_data.metadata_dict.popitem())

# video_names: SUB20_on_walk_19_2 
# participant_ID: SUB01
# pose_dict: ('SUB20_on_walk_19_2', array([[[ 0.  ,0. ,0. ],..]..])
# sensor_dict: ('SUB20_on_left', array([[[ 0. ,0. , 0. ],..]..])
# labels_dict: ('SUB20_on_walk_19_2', 2) 0/1/2 updrs score
# metadata_dict: ('SUB20_on_walk_19_2', array([[0, 0.8108, 1.0, 1.0, 0.6827]], dtype=float32))
# /media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/C3Dfiles_processed_new
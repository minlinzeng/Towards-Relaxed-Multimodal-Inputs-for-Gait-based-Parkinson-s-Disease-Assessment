import argparse
import os
import re
import pandas as pd
import numpy as np
import c3d
import xlrd
import csv
from .const_pd import H36M_FULL, PD
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# H36M_FULL = {
#     'B.TORSO': 0,
#     'L.HIP': 1,
#     'L.KNEE': 2,
#     'L.FOOT': 3,
#     'R.HIP': 4,
#     'R.KNEE': 5,
#     'R.FOOT': 6,
#     'C.TORSO': 7,
#     'U.TORSO': 8,
#     'NECK': 9,
#     'HEAD': 10,
#     'R.SHOULDER': 11,
#     'R.ELBOW': 12,
#     'R.HAND': 13,
#     'L.SHOULDER': 14,
#     'L.ELBOW': 15,
#     'L.HAND': 16
# }

H36M_CONNECTIONS_FULL = {
    (H36M_FULL['B.TORSO'], H36M_FULL['L.HIP']),
    (H36M_FULL['B.TORSO'], H36M_FULL['R.HIP']),
    (H36M_FULL['R.HIP'], H36M_FULL['R.KNEE']),
    (H36M_FULL['R.KNEE'], H36M_FULL['R.FOOT']),
    (H36M_FULL['L.HIP'], H36M_FULL['L.KNEE']),
    (H36M_FULL['L.KNEE'], H36M_FULL['L.FOOT']),
    (H36M_FULL['B.TORSO'], H36M_FULL['C.TORSO']),
    (H36M_FULL['C.TORSO'], H36M_FULL['U.TORSO']),
    (H36M_FULL['U.TORSO'], H36M_FULL['L.SHOULDER']),
    (H36M_FULL['L.SHOULDER'], H36M_FULL['L.ELBOW']),
    (H36M_FULL['L.ELBOW'], H36M_FULL['L.HAND']),
    (H36M_FULL['U.TORSO'], H36M_FULL['R.SHOULDER']),
    (H36M_FULL['R.SHOULDER'], H36M_FULL['R.ELBOW']),
    (H36M_FULL['R.ELBOW'], H36M_FULL['R.HAND']),
    (H36M_FULL['U.TORSO'], H36M_FULL['NECK']),
    (H36M_FULL['NECK'], H36M_FULL['HEAD'])
}


def convert_pd_h36m(sequence):
    new_keyponts = np.zeros((sequence.shape[0], 17, 3))
    new_keyponts[..., H36M_FULL['B.TORSO'], :] = (sequence[..., PD['L.ASIS'], :] +
                                             sequence[..., PD['R.ASIS'], :] +
                                             sequence[..., PD['L.PSIS'], :] +
                                             sequence[..., PD['R.PSIS'], :]) / 4
    new_keyponts[..., H36M_FULL['L.HIP'], :] = (sequence[..., PD['L.ASIS'], :] + 
                                           sequence[..., PD['L.PSIS'], :]) / 2
    new_keyponts[..., H36M_FULL['L.KNEE'], :] = sequence[..., PD['L.KNEE'], :]
    new_keyponts[..., H36M_FULL['L.FOOT'], :] = sequence[..., PD['L.ANKLE'], :]
    new_keyponts[..., H36M_FULL['R.HIP'], :] = (sequence[..., PD['R.ASIS'], :] + 
                                           sequence[..., PD['R.PSIS'], :]) / 2
    new_keyponts[..., H36M_FULL['R.KNEE'], :] = sequence[..., PD['R.KNEE'], :]
    new_keyponts[..., H36M_FULL['R.FOOT'], :] = sequence[..., PD['R.ANKLE'], :]
    new_keyponts[..., H36M_FULL['U.TORSO'], :] = (sequence[..., PD['C7'], :] + 
                                             sequence[..., PD['CLAV'], :]) / 2
    new_keyponts[..., H36M_FULL['C.TORSO'], :] = (sequence[..., PD['STRN'], :] + 
                                             sequence[..., PD['T10'], :]) / 2
    new_keyponts[..., H36M_FULL['R.SHOULDER'], :] = sequence[..., PD['R.SHO'], :]
    new_keyponts[..., H36M_FULL['R.ELBOW'], :] = (sequence[..., PD['R.EL'], :] + 
                                           sequence[..., PD['R.EM'], :]) / 2
    new_keyponts[..., H36M_FULL['R.HAND'], :] = (sequence[..., PD['R.WL'], :] + 
                                           sequence[..., PD['R.WM'], :]) / 2
    new_keyponts[..., H36M_FULL['L.SHOULDER'], :] = sequence[..., PD['L.SHO'], :]
    new_keyponts[..., H36M_FULL['L.ELBOW'], :] = (sequence[..., PD['L.EL'], :] + 
                                           sequence[..., PD['L.EM'], :]) / 2
    new_keyponts[..., H36M_FULL['L.HAND'], :] = (sequence[..., PD['L.WL'], :] + 
                                           sequence[..., PD['L.WM'], :]) / 2
    new_keyponts[..., H36M_FULL['NECK'], :] = new_keyponts[..., H36M_FULL['U.TORSO'], :] + [0.27, 57.48, 11.44]
    new_keyponts[..., H36M_FULL['HEAD'], :] = new_keyponts[..., H36M_FULL['U.TORSO'], :] + [-2.07, 165.23, 34.02]
    
    return new_keyponts

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='./PD_3D_motion-capture_data', type=str, help='Path to the input folder')
    args = parser.parse_args()
    return args

def rotate_around_z_axis(points, theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(points, R.T)

def visualize_sequence(seq, name):
    VIEWS = {
        "pd": {
            "best": (45, 20, 100),
            "best2": (0, 0, 0),
            "side": (90, 0, 90),
        },
        "tmp": {
            "best": (45, 20, 100),
            "side": (90, 0, 90),
        }
    }
    elev, azim, roll = VIEWS["pd"]["side"]
    # Apply the rotation to each point in the sequence
    for i in range(seq.shape[1]):
        seq[:, i, :] = rotate_around_z_axis(seq[:, i, :], roll)

    def update(frame):
        ax.clear()
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])

        # print(VIEWS[data_type][view_type])
        # ax.view_init(*VIEWS[data_type][view_type])
        elev, azim, roll = VIEWS["pd"]["best"]
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(f'Frame: {frame}')
        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        for connection in H36M_CONNECTIONS_FULL:
            start = seq[frame, connection[0], :]
            end = seq[frame, connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs)
        ax.scatter(x, y, z)

    print(f"Number of frames: {seq.shape[0]}")
    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frames = list(range(seq.shape[0]))
    ani = FuncAnimation(fig, update, frames=tqdm(frames, desc="Animating frames"), interval=50)
    ani.save(f'{name}.gif', writer='pillow')
    plt.close(fig)

def extract_sort_key(file_name):
    """
    Extracts sorting keys from the filename.
    Expected format: SUBXX_on/off_walk_X.c3d
    Returns a tuple: (subject number, condition order, walk number, start index)
    """
    match = re.search(r'SUB(\d+)_([Oo]n|[Oo]ff)_walk_(\d+)', file_name)
    if match:
        subject_number = int(match.group(1))  # Extracts XX from "SUBXX"
        condition = match.group(2).lower()    # "on" or "off"
        condition_order = 0 if condition == "on" else 1  # "On" first, "Off" second
        walk_number = int(match.group(3))     # Extracts X from "walk_X"
        return (subject_number, condition_order, walk_number)
    return (float('inf'), float('inf'), float('inf'))  # Place invalid names at the end

def read_pd(sequence_path):
    """
    Read points data from a .c3d file and create a sequence of selected frames.

    Parameters:
    sequence_path (str): The file path for the .c3d file.
    start_index (int): The frame index at which to start reading the data.
    step (int): The number of frames to skip between reads. A step of n reads every nth frame.

    Returns:
    numpy.ndarray: An array containing the processed sequence of points data from the .c3d file.

    """
    # sequence_path = './PD_3D_motion-capture_data/C3Dfiles/SUB09_off/SUB09_off_walk_6.c3d'
    reader = c3d.Reader(open(sequence_path, 'rb'))
    sequence = []
    cleaned_sequence = []
    total_frames = reader.frame_count 
    removed_frames = 0  
    
    for i, points, analog in reader.read_frames():
        sequence.append(points[:44, :3])  
        if np.any(np.all(points[:44, :3] == 0, axis=1)):   # Removed frames with corrupted joints
            removed_frames += 1
            continue
        cleaned_sequence.append(points[None, :44, :3])
        
    if len(cleaned_sequence) == 0:
        return np.array([]), 100.0, []  # If no frames exist, return 100% removal
        
    gap_dict = identify_gaps(sequence) # Identify gaps in the sequence   
    removal_rate = (removed_frames / total_frames) * 100
    cleaned_sequence = np.concatenate(cleaned_sequence)
    cleaned_sequence = convert_pd_h36m(cleaned_sequence)
    
    return cleaned_sequence, removal_rate, gap_dict

def identify_gaps(sequence):
    """
    Identify consecutive missing frame gaps in a processed sequence.

    Parameters:
    sequence (numpy.ndarray): The processed sequence.

    Returns:
    list: Lengths of consecutive missing frame gaps.
    """
    gap_dict = {}
    current_gap_length = 0
    gap_count = 0
    
    for idx, frame in enumerate(sequence):
        if np.any(np.all(frame == 0, axis=1)):
            current_gap_length += 1
            if current_gap_length == 1:
                gap_dict[gap_count] = f"{idx}-"  # Assign only if condition is met
        else:
            if current_gap_length > 0:
                gap_dict[gap_count] += f"{idx}:{current_gap_length}"
                gap_count += 1
                current_gap_length = 0  # Reset counter
    
    if current_gap_length > 0:  # Save last gap if still counting
        gap_dict[gap_count] += f"{len(sequence)}:{current_gap_length}"
 
    return gap_dict

def extract_grf_data(grf_root_folder, output_folder):
    """
    1) Iterates over all GRF CSVs in `grf_root_folder`.
    2) Extracts (x,y,z) trials from CSV columns.
    3) Groups trials by subject & foot side.
       - Left foot -> shape: (101, left_trial_count, 3)
       - Right foot -> shape: (101, right_trial_count, 3)
    4) Saves to `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)
    subject_data = {}

    for subj_folder in os.listdir(grf_root_folder):  
        subj_path = os.path.join(grf_root_folder, subj_folder)
        if not os.path.isdir(subj_path):
            continue  # Skip if it's not a directory

        subject_id = subj_folder  # e.g., "SUB01"
        # Initialize dictionary for this subject.
        if subject_id not in subject_data:
            subject_data[subject_id] = {'on_left': [], 'on_right':[], 'off_left':[], 'off_right':[]}
        
        for condition in ["ON", "OFF"]:  
            grf_folder = os.path.join(subj_path, condition, "GRF")
            if not os.path.exists(grf_folder):
                continue  # Skip if no GRF folder found

            # Iterate over all CSV files in this GRF folder.
            for csv_file in os.listdir(grf_folder):
                if not csv_file.endswith('.csv'):
                    continue

                csv_path = os.path.join(grf_folder, csv_file)
                # Extract subject ID and foot side from filename
                foot_side = 'left' if 'left' in csv_file.lower() else 'right' if 'right' in csv_file.lower() else 'sum_cycles'

                if foot_side != 'sum_cycles':
                    # Read CSV (ignoring first row and column if it's "Gait cycle")
                    df = pd.ExcelFile(csv_path)
                    df = pd.read_excel(df, sheet_name=df.sheet_names[0])
                    if 'gait' in df.columns[0].lower():
                        df = df.iloc[1:, 1:]  # Drop the first row and column

                    # Extract (x,y,z) columns per trial
                    num_cols = df.shape[1]
                    trial_arrays = []
                    for start_col in range(0, num_cols, 3):
                        end_col = start_col + 3
                        if end_col > num_cols:
                            break  # Ensure correct shape
                        trial_data = df.iloc[:, start_col:end_col].to_numpy()  # shape (101,3)
                        trial_arrays.append(trial_data)
                        
                    # Append trials to the corresponding foot side.
                    subject_data[subject_id][condition.lower()+'_'+foot_side].extend(trial_arrays)


    # Combine & save for each subject
    for subj_id, foot_dict in subject_data.items():
        on_left_trials = foot_dict['on_left']  # List of arrays, each (101,3)
        on_right_trials = foot_dict['on_right']  # List of arrays, each (101,3)
        off_left_trials = foot_dict['off_left']  # List of arrays, each (101,3)
        off_right_trials = foot_dict['off_right']  # List of arrays, each (101,3)

        # Concatenate along new axis (trial axis)
        on_left_cat = (np.concatenate([trial[:, np.newaxis, :] for trial in on_left_trials], axis=1)
                    if on_left_trials else np.zeros((101, 0, 3)))
        on_right_cat = (np.concatenate([trial[:, np.newaxis, :] for trial in on_right_trials], axis=1)
                     if on_right_trials else np.zeros((101, 0, 3)))
        off_left_cat = (np.concatenate([trial[:, np.newaxis, :] for trial in off_left_trials], axis=1)
                    if off_left_trials else np.zeros((101, 0, 3)))
        off_right_cat = (np.concatenate([trial[:, np.newaxis, :] for trial in off_right_trials], axis=1)
                     if off_right_trials else np.zeros((101, 0, 3)))
        
        # Save separate npy files for left and right foot.
        on_left_out_path = os.path.join(output_folder, f"{subj_id}_on_left.npy")
        on_right_out_path = os.path.join(output_folder, f"{subj_id}_on_right.npy")
        off_left_out_path = os.path.join(output_folder, f"{subj_id}_off_left.npy")
        off_right_out_path = os.path.join(output_folder, f"{subj_id}_off_right.npy")
        np.save(on_left_out_path, on_left_cat)
        np.save(on_right_out_path, on_right_cat)
        np.save(off_left_out_path, off_left_cat)
        np.save(off_right_out_path, off_right_cat)
        print(f"[GRF] Saved {on_left_out_path} => shape {on_left_cat.shape}")
        print(f"[GRF] Saved {on_right_out_path} => shape {on_right_cat.shape}")
        print(f"[GRF] Saved {off_left_out_path} => shape {off_left_cat.shape}")
        print(f"[GRF] Saved {off_right_out_path} => shape {off_right_cat.shape}")

def record_processed_sequences(cleaned_sequence, removal_rate, gap_dict, sequence_path):
    """records each file's stats in a dictionary for csv writing."""
    csv_rows = {}
    sequence_length = len(cleaned_sequence)
    if sequence_length == 0:
        csv_rows = {
            "file names": f"{os.path.basename(sequence_path)[:-4]}",
            "sequence length": 0,
            "removal_rate": 'NA',
            "gaps info": 'NA'
        }
    else:
        gaps_info = f"gaps: {gap_dict.items()}" if gap_dict else "0 gaps"
        csv_rows = {
            "file names": f"{os.path.basename(sequence_path)[:-4]}",
            "sequence length": sequence_length,
            "removal_rate": removal_rate,
            "gaps info": gaps_info
        }
    return csv_rows

def main():
    """
    Preprocesses C3D files for Parkinson's disease dataset.
    This function takes the input path, reads C3D files, processes them, and saves the processed sequences as numpy arrays.
    """
    args = parse_args()
    csv_rows = []
    file_list = []

    input_path_c3dfiles = os.path.join(args.input_path, 'C3Dfiles')
    # output_path_c3dfiles = os.path.join(args.input_path, 'C3Dfiles_processed_new')
    output_path_c3dfiles = os.path.join(args.input_path, 'C3Dfiles_cleaned_sequences')

    if not os.path.exists(input_path_c3dfiles):
        raise FileNotFoundError(f"Input folder '{input_path_c3dfiles}' not found.")
    os.makedirs(output_path_c3dfiles, exist_ok=True)
    
    for root, dirs, files in os.walk(input_path_c3dfiles):
        for file in files:
            if file.endswith('.c3d') and "walk" in file and file.startswith("SUB"):
                file_list.append(os.path.join(root, file))
                
    file_list.sort(key=lambda x: extract_sort_key(os.path.basename(x)))
    for sequence_path in file_list:
        file_name = os.path.basename(sequence_path)[:-4]         
        try:
            cleaned_sequence, removal_rate, gap_dict = read_pd(sequence_path) # remove corrupted frames
            csv_rows.append(record_processed_sequences(cleaned_sequence, removal_rate, gap_dict, sequence_path))
            file_name = os.path.join(output_path_c3dfiles, file_name+'.npy')
            if len(cleaned_sequence) > 0:
                # np.save(file_name, cleaned_sequence)
                pass
            # np.save(file_name, cleaned_sequence)
        except Exception as e:
            print(f"Error reading {sequence_path}: {str(e)}")
    
    # csv_file_path = os.path.join(output_path_c3dfiles, "processed_sequences.csv")
    # with open(csv_file_path, mode='w', newline='') as csvfile:
    #     fieldnames = ["file names", "sequence length", "removal_rate", "gaps info"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for row in csv_rows:
    #         writer.writerow(row)
    
    # grf_root_folder = os.path.join(args.input_path, 'Gait cycle')
    # grf_output_folder = os.path.join(args.input_path, 'GRF_processed')
    # extract_grf_data(grf_root_folder, grf_output_folder)


# if __name__ == "__main__":
#     main()
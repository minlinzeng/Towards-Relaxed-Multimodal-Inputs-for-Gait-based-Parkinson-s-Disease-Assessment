import torch
import random
import numpy as np
import copy


H36M_FULL = {
    'B.TORSO': 0,
    'L.HIP': 1,
    'L.KNEE': 2,
    'L.FOOT': 3,
    'R.HIP': 4,
    'R.KNEE': 5,
    'R.FOOT': 6,
    'C.TORSO': 7,
    'U.TORSO': 8,
    'NECK': 9,
    'HEAD': 10,
    'R.SHOULDER': 11,
    'R.ELBOW': 12,
    'R.HAND': 13,
    'L.SHOULDER': 14,
    'L.ELBOW': 15,
    'L.HAND': 16
}

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


def rotate_around_z_axis(points, theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(points, rotation.T)


def visualize_sequence(seq, name):
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    seq = seq.copy()
    for i in range(seq.shape[1]):
        seq[:, i, :] = rotate_around_z_axis(seq[:, i, :], 90)

    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))
    aspect_ratio = [max_x - min_x, max_y - min_y, max_z - min_z]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.view_init(elev=45, azim=20)
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(f'Frame: {frame}')

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        for connection in H36M_CONNECTIONS_FULL:
            start = seq[frame, connection[0], :]
            end = seq[frame, connection[1], :]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
        ax.scatter(x, y, z)

    print(f"Number of frames: {seq.shape[0]}")
    animation = FuncAnimation(fig, update, frames=seq.shape[0], interval=1)
    animation.save(f'{name}.gif', writer='pillow')
    plt.close(fig)


class MirrorReflection:
    """
    Do horizontal flipping for each frame of the sequence.
    
    Args:
      format (str): Skeleton format. By default it expects it to be h36m (as motion encoders are mostly trained on that)
    """

    def __init__(self, format='h36m', data_dim=2):
        if format == 'h36m':
            self.left = [14, 15, 16, 1, 2, 3]
            self.right = [11, 12, 13, 4, 5, 6]
        else:
            raise NotImplementedError("Skeleton format is not supported.")
        self.data_dim = data_dim
    
    def __call__(self, sample):
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence)
        if self.data_dim == 3:
            merge_last_dim = 0 
            if sequence.ndim == 2:
                sequence = sequence.view(-1, 17, 3)  # Reshape sequence back to N x 17 x 3
                merge_last_dim = 1   
        mirrored_sequence = sequence.clone()
        mirrored_sequence[:, :, 0] *= -1
        mirrored_sequence[:, self.left + self.right, :] = mirrored_sequence[:, self.right + self.left, :]
        
        if self.data_dim == 3 and merge_last_dim: # Reshape sequence back to N x 51
                N = np.shape(mirrored_sequence)[0]
                mirrored_sequence = mirrored_sequence.reshape(N, -1)
        return {
            'encoder_inputs': mirrored_sequence,
            'label': label,
            'labels_str': labels_str
        }


class RandomRotation:
    """
    Rotate randomly all the joints in all the frames.

    Args:
       min_rotate (int): Minimum degree of rotation angle.
       max_rotate (int): Maximum degree of rotation angle.
    """

    def __init__(self, min_rotate, max_rotate, data_dim=2):
        self.min_rotate, self.max_rotate = min_rotate, max_rotate
        self.data_dim = data_dim
    
    def _create_3d_rotation_matrix(self, axis, rotation_angle):
        theta = rotation_angle * (torch.pi / 180)
        if axis == 0:  # x-axis
            rotation_matrix = torch.tensor([[1, 0, 0],
                            [0, torch.cos(theta), torch.sin(theta)],
                            [0, -torch.sin(theta), torch.cos(theta)]])
        elif axis == 1:  # y-axis
            rotation_matrix = torch.tensor([[torch.cos(theta), 0, -torch.sin(theta)],
                            [0, 1, 0],
                            [torch.sin(theta), 0, torch.cos(theta)]])
        elif axis == 2:  # z-axis
            rotation_matrix = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                            [-torch.sin(theta), torch.cos(theta), 0],
                            [0, 0, 1]])
        return rotation_matrix

    def _create_rotation_matrix(self):
        rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate, self.max_rotate)
        theta = rotation_angle * (torch.pi / 180)

        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                        [torch.sin(theta), torch.cos(theta)]])
        return rotation_matrix
    
    def __call__(self, sample):
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence)

        if self.data_dim == 2:
            rotation_matrix = self._create_rotation_matrix()

            sequence_has_confidence_score = sequence.shape[-1] == 3
            if sequence_has_confidence_score:
                rotated_sequence = sequence[..., :2] @ rotation_matrix
                rotated_sequence = torch.cat((rotated_sequence, sequence[..., 2].unsqueeze(-1)), dim=-1)
            else:
                rotated_sequence = sequence @ rotation_matrix
        else: 
            merge_last_dim = 0 
            if sequence.ndim == 2:
                sequence = sequence.view(-1, 17, 3)  # Reshape sequence back to N x 17 x 3
                merge_last_dim = 1
            rotated_sequence = sequence.clone()
            total_axis = [0, 1, 2]
            main_axis = random.randint(0, 2)
            for axis in total_axis:
                if axis == main_axis:
                    rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate, self.max_rotate)
                    rotation_matrix = self._create_3d_rotation_matrix(axis, rotation_angle)
                else:
                    rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate/10, self.max_rotate/10)
                    rotation_matrix = self._create_3d_rotation_matrix(axis, rotation_angle)
                rotated_sequence = rotated_sequence @ rotation_matrix
            if merge_last_dim: # Reshape sequence back to N x 51
                N = np.shape(rotated_sequence)[0]
                rotated_sequence = rotated_sequence.reshape(N, -1)
            
        return {
            'encoder_inputs': rotated_sequence,
            'label': label,
            'labels_str': labels_str
        }
    

class RandomNoise:
    """
    Adds noise randomly to each join separately from normal distribution.
    """
    def __init__(self, mean=0, std=0.01, data_dim=2):
        self.mean = mean
        self.std = std
        self.data_dim = data_dim

    def __call__(self, sample):
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence)
        noise = torch.normal(self.mean, self.std, size=sequence.shape)
        noise_sequence = sequence + noise

        return {
            'encoder_inputs': noise_sequence,
            'label': label,
            'labels_str': labels_str
        }
        
        
class axis_mask:
    def __init__(self, data_dim=3):
        self.data_dim = data_dim
    
    def Zero_out_axis(self, sequence):
        axis_next = random.randint(0, self.data_dim-1) 
        temp = sequence.clone()
        T, J, C = sequence.shape
        x_new = torch.zeros(T, J, device=temp.device)
        temp[:, :, axis_next] = x_new
        return temp
    
    def __call__(self, sample):
        
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
                sequence = torch.from_numpy(sequence)
                
        if self.data_dim > 2 :
            if self.data_dim == 3:
                merge_last_dim = 0 
                if sequence.ndim == 2:
                    sequence = sequence.view(-1, 17, 3)  # Reshape sequence back to N x 17 x 3
                    merge_last_dim = 1
            masked_sequence = self.Zero_out_axis(sequence)
            
            if self.data_dim == 3 and merge_last_dim: # Reshape sequence back to N x 51
                    N = np.shape(masked_sequence)[0]
                    masked_sequence = masked_sequence.reshape(N, -1)
        
            return {
                    'encoder_inputs': masked_sequence,
                    'label': label,
                    'labels_str': labels_str
                }
        else:
            return {
                    'encoder_inputs': sequence,
                    'label': label,
                    'labels_str': labels_str
                }


class PoseSequenceAugmentation:
    def __init__(self, params):
        self.augmentation_methods = {
            "mirror_reflection": self.mirror_reflection,
            "joint_dropout": self.joint_dropout,
            "random_rotation": self.random_rotation,
            "random_translation": self.random_translation
        }
        self.params = params

    def augment_data(self, raw_data, augmentation_list, visualize_only=False):
        if "random_translation" in augmentation_list:
            self.estimate_translation_range(raw_data.pose_dict)

        augmented_data_dict = {"pose_dict": {}, "labels_dict": {}}

        augmented_video_names = []
        for video_name, pose_sequence in raw_data.pose_dict.items():
            augmented_sequences = {}

            for augmentation_name in augmentation_list:
                if augmentation_name in self.augmentation_methods:
                    augmented_sequence = self.augmentation_methods[augmentation_name](pose_sequence)
                    augmented_sequences[augmentation_name] = augmented_sequence
                    if visualize_only:
                        visualize_sequence(pose_sequence, video_name + '_org')
                        visualize_sequence(augmented_sequence, f"{video_name}_{augmentation_name}")
                else:
                    print(f"Warning: Unknown augmentation technique '{augmentation_name}'")

            if visualize_only:
                exit()

            for augmentation_name, augmented_sequence in augmented_sequences.items():
                augmented_video_name = f"{video_name}_{augmentation_name}"
                augmented_video_names.append(augmentation_name)

                augmented_data_dict["pose_dict"][augmented_video_name] = augmented_sequence
                augmented_data_dict["labels_dict"][augmented_video_name] = raw_data.labels_dict[
                    video_name]

        return self.update_datareader(raw_data, augmented_data_dict, augmented_video_names)

    @staticmethod
    def update_datareader(raw_data, augmented_data_dict, augmented_video_names):
        raw_data_augmented = copy.deepcopy(raw_data)
        raw_data_augmented.labels = raw_data_augmented.labels + list(
            augmented_data_dict['labels_dict'].values())
        raw_data_augmented.video_names = raw_data_augmented.video_names + augmented_video_names
        raw_data_augmented.labels_dict.update(augmented_data_dict['labels_dict'])
        raw_data_augmented.pose_dict.update(augmented_data_dict['pose_dict'])
        return raw_data_augmented

    @staticmethod
    def mirror_reflection(pose_sequence):
        mirrored_sequence = pose_sequence.copy()
        left = [4, 5, 6, 10, 11, 12]
        right = [7, 8, 9, 13, 14, 15]
        mirrored_sequence[:, :, 0] *= -1
        mirrored_sequence[:, left + right, :] = mirrored_sequence[:, right + left, :]
        return mirrored_sequence

    @staticmethod
    def joint_dropout(pose_sequence, dropout_prob):
        dropout_mask = np.random.choice([0, 1], size=pose_sequence.shape[1], p=[dropout_prob, 1 - dropout_prob])
        dropped_sequence = pose_sequence * dropout_mask
        return dropped_sequence

    def random_rotation(self, pose_sequence):
        rotation_angles = np.random.uniform(self.params['rotation_range'][0], self.params['rotation_range'][1], size=3)
        rotation_matrix = self.rotation_matrix(rotation_angles)
        rotated_sequence = np.matmul(pose_sequence, rotation_matrix)
        return rotated_sequence

    def random_translation(self, pose_sequence):
        noise_scale = 0
        translation = np.random.uniform(self.translation_range[0], self.translation_range[1], size=3)
        noise = np.random.normal(scale=noise_scale, size=pose_sequence.shape)
        translated_sequence = pose_sequence + translation + noise
        return translated_sequence

    def estimate_translation_range(self, pose_dict):
        min_values = np.min([np.min(pose) for pose in pose_dict.values()])
        max_values = np.max([np.max(pose) for pose in pose_dict.values()])
        overall_range = max_values - min_values
        self.translation_range = (-self.params['translation_frac'] * overall_range,
                                  self.params['translation_frac'] * overall_range)

    def estimate_noise_scale(pose_dict):
        min_values = np.min([np.min(pose) for pose in pose_dict.values()])
        max_values = np.max([np.max(pose) for pose in pose_dict.values()])
        overall_range = max_values - min_values
        noise_scale = 0.1 * overall_range
        return noise_scale

    @staticmethod
    def rotation_matrix(angles):
        radians = angles * (np.pi / 180)
        alpha, beta, gamma = radians
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))
        return rotation_matrix


def walkid_to_AMBID(cur_walk_id):
    raw_id = cur_walk_id
    if raw_id >= 60:
        id = raw_id - 3
    else:
        id = raw_id - 2
    return id


def get_AMBID_from_Videoname(path_file):
    AMBID = walkid_to_AMBID(int(path_file[24:26]))
    AMBID = 'AMB' + str(AMBID).zfill(2)
    return AMBID


def extract_unique_subs(dataset):
    if dataset is None:
        return []
    unique_subs = set()
    for name in dataset.video_names:
        sub = name.split('_')[0]
        unique_subs.add(sub)
    return list(unique_subs)


def count_labels(dataset, all_labels):
    label_counts = {lbl: 0 for lbl in all_labels}
    if dataset is not None:
        labels, counts = np.unique(dataset.labels, return_counts=True)
        label_counts.update(dict(zip(labels, counts)))
    return label_counts

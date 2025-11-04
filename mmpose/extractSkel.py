from mmpose.apis import MMPoseInferencer
import multiprocessing as mp
import json
import os
import numpy as np
import cv2

video_folder = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/Videos/'
vis_out_3d_dir = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/lifted/'
pred_out_dir = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/predictions/'
frame_skip = 1
num_workers = 6  # number of processes (tune based on your CPU/GPU)
device = 'cuda:0'  # GPU
os.makedirs(pred_out_dir, exist_ok=True)

# Utility to get total frames
def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

# Worker function
def process_videos(video_list, worker_id):
    log_dir = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/logs/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'worker_{worker_id}.log')
    log_file = open(log_path, 'w')
    
    def log(message):
        log_file.write(message + '\n')
        log_file.flush()
    log(f"Worker {worker_id} started. PID: {os.getpid()}.")
    
    inferencer = MMPoseInferencer(pose3d='human3d', device=device)
    for each in video_list:
        if not each.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        
        log(f"Processing {each}...")
        video_path = os.path.join(video_folder, each)
        video_name = os.path.splitext(each)[0]
        output_json = os.path.join(pred_out_dir, video_name + '_3d_predictions.json')
        total_frames = get_total_frames(video_path)

        result_generator = inferencer(
            video_path,
            show=False,
            vis_out_dir=vis_out_3d_dir,
            pred_out_dir=None,
            num_instances=1,
        )

        frame_idx = 0
        skipped_frames = 0  # 🔥 count skipped frames
        processed_frames = 0  # 🔥 count actually processed frames
        results = []

        for result in result_generator:
            frame_idx += 1
            
            if not result['predictions']:  # if no person detected
                skipped_frames += 1
                continue  # 🔥 skip this frame
            
            if frame_idx % frame_skip != 0:
                continue
            processed_frames += 1
            results.append(result)
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                percent = frame_idx / total_frames * 100
                log(f"Progress: {frame_idx}/{total_frames} ({percent:.2f}%)")

        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        log(f"Finished {each}")
        log(f"Frames processed: {processed_frames}")
        log(f"Frames skipped (no detection): {skipped_frames}")
        log(f"Total frames in video: {total_frames}")
        log(f"Effective frame usage rate: {processed_frames / total_frames * 100:.2f}%\n")
    log_file.close()
# Main multi-processing setup
 
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    all_videos = [v for v in os.listdir(video_folder) if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    existing_outputs = [f for f in os.listdir(pred_out_dir) if f.endswith('_3d_predictions.json')]
    existing_video_names = set(os.path.splitext(f.replace('_3d_predictions', ''))[0] for f in existing_outputs)
    
    videos_to_process = []
    for video in all_videos:
        video_name = os.path.splitext(video)[0]
        if video_name not in existing_video_names:
            videos_to_process.append(video)
        else:
            print(f"Skipping {video}, already processed.")

    print(f"{len(videos_to_process)} videos to process.")
    
    # Split videos among workers
    split_videos = [videos_to_process[i::num_workers] for i in range(num_workers)]

    processes = []
    for worker_id, worker_videos in enumerate(split_videos):
        p = mp.Process(target=process_videos, args=(worker_videos, worker_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All videos processed by multi-process!")


# # Prepare where to save skeleton data
# os.makedirs(vis_out_dir, exist_ok=True)
# json_save_path = os.path.join(vis_out_dir, video_path.split('/')[-1][:-4]+'.json')

# # Run inference
# result_generator = inferencer(
#     video_path,
#     show=False,
#     vis_out_dir=vis_out_dir
# )

# # Collect all skeletons
# all_keypoints = []

# for frame_idx, result in enumerate(result_generator):
#     frame_info = {
#         'frame_id': frame_idx,
#         'instances': []
#     }
#     """
#     l/r hips (11, 12)
#     knees (13, 14)
#     ankles (15, 16)
#     """
#     predictions = result['predictions']
#     for instance in predictions:
#         instance = instance[0]
#         frame_info['instances'].append({
# 			'keypoints': instance['keypoints'][11:17],  # already plain lists
# 			'keypoint_scores': [float(x) for x in instance['keypoint_scores'][11:17]],
# 			# 'bbox': [float(x) for x in instance['bbox'][0]] if instance.get('bbox') else None,
# 			# 'bbox_score': float(instance.get('bbox_score')) if instance.get('bbox_score') else None
# 		})
#     all_keypoints.append(frame_info)

# # Save keypoints to JSON
# with open(json_save_path, 'w') as f:
#     json.dump(all_keypoints, f, indent=4)


# config_file = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/mmpose/video-pose-lift_tcn-27frm-semi-supv-cpn-ft_8xb64-200e_h36m.py'
# checkpoint_file = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/mmpose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth'

# Initialize the 3D lifter inferencer.
# inferencer_3d = MMPoseInferencer(pose3d=config_file, pose3d_weights=checkpoint_file)

# print(f"Skeleton coordinates saved to: {json_save_path}")
# print(f"Visualization video saved under: {vis_out_dir}")
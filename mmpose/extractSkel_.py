#!/usr/bin/env python3
"""
Re-runnable multi-process 3D pose processing script for videos.

Each run:
  - Checks the video folder for videos that do not have corresponding _3d_predictions.json files.
  - Splits the work among several workers.
  - Each worker processes its assigned video(s) (ideally one video per process) then stops.
  - Once all workers finish, if there remain unprocessed videos, the script restarts itself.
  - This continues until all videos are processed.
"""

import os
import sys
import json
import cv2
import numpy as np
import multiprocessing as mp
from mmpose.apis import MMPoseInferencer

# ------------------ CONFIGURATION ------------------
# Folder paths (update these paths for your environment)
video_folder = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/lifted/'
vis_out_3d_dir = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/lifted_new/'
pred_out_dir = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/predictions/'
log_dir = '/media/hdd/minlin/MotionEncoders_parkinsonism_benchmark/PD_3D_motion-capture_data/turn-in-place/logs/'

# Processing parameters
frame_skip = 0
num_workers = 6  # Tune based on your hardware (here, 40 CPU threads, 1 GPU)
device = 'cuda:0'  # GPU device

# Ensure necessary output folders exist
os.makedirs(pred_out_dir, exist_ok=True)
os.makedirs(vis_out_3d_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# ------------------ HELPER FUNCTIONS ------------------
def get_total_frames(video_path):
    """Return the total number of frames in the video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def check_unfinished_videos():
    """Return a list of video filenames in video_folder that do NOT have a corresponding _3d_predictions.json file in pred_out_dir."""
    all_videos = [v for v in os.listdir(video_folder) if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    existing_outputs = [f for f in os.listdir(pred_out_dir) if f.endswith('_3d_predictions.json')]
    existing_video_names = set(os.path.splitext(f.replace('_3d_predictions', ''))[0] for f in existing_outputs)
    unfinished = []
    for video in all_videos:
        video_name = os.path.splitext(video)[0]
        if video_name not in existing_video_names:
            unfinished.append(video)
    return unfinished


def process_one_video(video_name, worker_id, log):
    """
    Process a single video using the MMPose 3D lifter inferencer.
    Logs progress to the provided log function.
    """
    video_path_full = os.path.join(video_folder, video_name)
    base_name = os.path.splitext(video_name)[0]
    output_json = os.path.join(pred_out_dir, base_name + '_3d_predictions.json')
    total_frames = get_total_frames(video_path_full)

    log(f"Worker {worker_id}: Processing {video_name} (Total frames: {total_frames})")
    # Initialize the 3D lifter inferencer
    inferencer = MMPoseInferencer(pose3d='human3d', device=device)

    result_generator = inferencer(
        video_path_full,
        show=False,
        vis_out_dir=vis_out_3d_dir,
        pred_out_dir=None,
        num_instances=1,  # optional: limit to one instance per frame
        tracking_thr=0.9,
        kpt_thr=0.9
    )

    frame_idx = 0
    skipped_frames = 0
    processed_frames = 0
    results = []
    for result in result_generator:
        frame_idx += 1

        # If no prediction is returned, skip this frame.
        if not result.get('predictions'):
            skipped_frames += 1
            continue

        # if frame_idx % frame_skip != 0:
        #     continue

        processed_frames += 1
        results.append(result)

        if frame_idx % 150 == 0 or frame_idx == total_frames:
            percent = frame_idx / total_frames * 100
            log(f"Worker {worker_id}: {video_name} progress: {frame_idx}/{total_frames} ({percent:.2f}%)")

    # Save the prediction results to JSON.
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    log(f"Worker {worker_id}: Finished {video_name}")
    log(f"Worker {worker_id}: Frames processed: {processed_frames}, skipped: {skipped_frames}, total: {total_frames}")
    log(f"Worker {worker_id}: Effective usage rate: {processed_frames / total_frames * 100:.2f}%\n")


def process_videos(video_list, worker_id):
    """Worker function: Process a list of videos (one per run) and then exit."""
    # Open a log file specific for this worker.
    log_path = os.path.join(log_dir, f'worker_{worker_id}.log')
    with open(log_path, 'a') as log_file:
        def log(msg):
            log_file.write(msg + '\n')
            log_file.flush()
            print(f"[Worker {worker_id}] {msg}")  # Also print to console

        log(f"Started. PID: {os.getpid()}.")

        # Process one video at a time; exit after the first successful video.
        for video in video_list:
            try:
                process_one_video(video, worker_id, log)
                # After one video is processed, exit the worker.
                log(f"Exiting after processing one video: {video}")
                break
            except Exception as e:
                log(f"Error processing {video}: {e}. Trying next video...")
                continue


# ------------------ MAIN PROCESSING & RE-RUN LOGIC ------------------
if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn' (required for CUDA)
    mp.set_start_method('spawn', force=True)

    unfinished_videos = check_unfinished_videos()
    num_unfinished = len(unfinished_videos)
    print(f"{num_unfinished} videos to process.")
    if num_unfinished == 0:
        print("✅ All videos processed. Exiting.")
        sys.exit(0)

    # Split the unfinished videos among workers.
    split_videos = [unfinished_videos[i::num_workers] for i in range(num_workers)]
    processes = []
    for worker_id, video_list in enumerate(split_videos):
        if len(video_list) == 0:
            continue
        p = mp.Process(target=process_videos, args=(video_list, worker_id))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    # print("Batch finished. Checking for remaining videos...")

    # After one batch, if there are still unprocessed videos, re-run this script.
    # remaining = check_unfinished_videos()
    # if remaining:
    #     print(f"{len(remaining)} videos remain unprocessed. Restarting script...")
    #     # Relaunch itself using subprocess; this will re-check and process remaining videos.
    #     subprocess.call([sys.executable] + sys.argv)
    # else:
    print("✅ All videos processed. Exiting.")
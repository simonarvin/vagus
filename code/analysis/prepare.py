import cv2
import sys
import numpy as np
import os
from arguments import Arguments
from constants import width, height

TYPE = "prepare"

args = Arguments(type = TYPE).args()

dlc = args.dlc
if dlc:
    try:
        import deeplabcut
    except ImportError:
        print("could not import deeplabcut.")
        print("run DLC anaconda env (eg: conda activate DLC-GPU)")

config_path = f"{args.dlcpath}/config.yaml"
dlc_plot = args.dlcplot
skip = args.skip
fps = args.fps
root_path = args.path

print(f"fps: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if "data" in root_path[-5:]:
    root_dirs = [rootdir for rootdir in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, rootdir))]
else:
    root_dirs = [""]

for root_dir in root_dirs:
    path = f"{root_path}/{root_dir}"

    subdirs = [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir)) and subdir != "calibration"]

    for subdir in subdirs:
        trial_dir = f"{path}/{subdir}"

        with open(f"{trial_dir}/datalog.log", "r") as log_file:
            log_lines = np.array(log_file.readlines(), dtype=np.float64)
        frames = len(log_lines)
        print(f"datalog file opened, id = {subdir}")

        save_dir = f"{trial_dir}/vids"
        try:
            os.mkdir(save_dir)
            print("vids folder created")
        except FileExistsError:
            print("vids folder already exists")
        print(f"saving videos in {save_dir}")

        cam_dirs = [ cam_dir for cam_dir in os.listdir(trial_dir) if os.path.isdir(os.path.join(trial_dir, cam_dir)) and "cam" in cam_dir ]
        print(f"cams detected: {cam_dirs}, n = {len(cam_dirs)}")
        videos = []
        for cam_dir in cam_dirs:
            if "cam" in cam_dir:
                cam_full_dir = f"{trial_dir}/{cam_dir}"
                
                print(f"processing {cam_dir}")
                vid_file = f"{save_dir}/{subdir}_{cam_dir}.mp4"
                videos.append(vid_file.replace("\\", "/"))
                if os.path.exists(vid_file) and skip:
                    print(f"{cam_dir} vid exists; skipping..")
                    continue
                else:
                    out = cv2.VideoWriter(vid_file, fourcc, fps, (width, height))

                for frame in np.arange(frames):
                    if frame % 100 == 0: #visualize progress
                        progress = round(20 * frame/frames)
                        print(f"[{'#' * progress + (20 - progress) * '.'}] {round(frame/frames * 100, 1)}%", end = "\r", flush = True)

                    img = cv2.imread(f"{cam_full_dir}/{frame}.jpg") #load frame
                    out.write(img.astype(np.uint8)) #write frame to output video

                out.release()
                print(f"\n(success) {cam_dir} saved")

        if dlc: #deeplabcut, machine learning body tracking
            print("applying DLC analysis on videos..")
            deeplabcut.analyze_videos(config_path, videos, videotype='mp4', save_as_csv=True, destfolder=save_dir)
            print("filtering DLC predictions..")
            deeplabcut.filterpredictions(config_path, videos, videotype='.mp4', filtertype= 'arima', destfolder=save_dir, ARdegree=5, MAdegree=2)
            if dlc_plot:
                print("plotting dlc data")
                deeplabcut.create_labeled_video(config_path, videos, videotype='.mp4', destfolder = save_dir, filtered = True)

print(f"(success) {TYPE} pipeline complete")

import cv2
import numpy as np
import os
import pickle
from moviepy.editor import *


CATEGORIES =["boxing","handwaving","running"]
os.makedirs("class", exist_ok=True)
# parameter declaration for computing dense optical flow features using
farneback_params = dict(winsize = 20, iterations=1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1, pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)
# read the video file
folder="./VideoDataset"
for category in os.listdir(folder):
    folder_path = os.path.join('./Videodataset' +'/'+ category)
    video_files=[]
    for filename in os.listdir(folder_path):
        filepath = os.path.join('./Videodataset' +'/'+ category+'/'+filename)
        clip = VideoFileClip(filepath)
        current_video= []
        prev_frame = None
        for frames in clip.iter_frames():
            gray_frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
            if prev_frame is not None:
                # Calculate optical flow.
                flows = cv2.calcOpticalFlowFarneback(prev_frame,gray_frames,**farneback_params)
                frame = []
                for r in range(120):
                    if r % 10 != 0:
                        continue
                    for c in range(160):
                        if c % 10 != 0:
                            continue
                        frame.append(flows[r,c,0])
                        frame.append(flows[r,c,1])
                frame = np.array(frame)
                current_video.append(frame)
            prev_frame = gray_frames
        # append the feature of each video file into the main list
        video_files.append({
            "filename": filename,
            "category": category,
            "features": current_video
        })
        clip.close()
    pickle.dump(video_files, open("class/optflow_%s.p" % category, "wb"))
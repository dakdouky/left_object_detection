import copy
import os
from turtle import back

import cv2
import numpy as np

from utils.visualize import plot_batch

import sys

import torch
import time
import torchvision.transforms as tvtf


dim = (320, 240)

cuda = True

dev = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = dev
print("Using GPU: ", dev) 
torch.cuda.empty_cache()


from modules.foreground_detector import ForegroundDetector 

model_name = 'saved_models/model_epoch_FCS.mdl'

FGD = ForegroundDetector(dim = dim)
FGD.laod_model(model_name)


vid = "Easy"
vid_extention = "mp4"
prefix = ""
max_record = 1000

background = cv2.imread(f"backgrounds/{vid}_bg/{vid}_background_{max_record}.jpg")
if not os.path.exists(f"infer_results/{vid}/"):
    os.mkdir(f"infer_results/{vid}/")
if not os.path.exists(f"infer_results/{vid}_masks/"):
    os.mkdir(f"infer_results/{vid}_masks/")
vid_path = f"videos/{prefix}{vid}.{vid_extention}"
out_path = f"videos/{prefix}{vid}_output.mp4"

if background is None: 
    print(f"Can't load background at: inference/background/{vid}.jpg")

vid_cap = cv2.VideoCapture(vid_path)
assert vid_cap.isOpened(), "Error opening video stream or file: {}".format(vid_path)

frame_id = 0
print(f"Processing video: {vid} ...")
h, w, c = background.shape
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(f'inference/output/{vid}.mp4', fourcc, 8.0, (4000,4000))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
hmap_out = cv2.VideoWriter(f'inference/output/{vid}_hmap.mp4', fourcc, 20.0, (320, 240))

while vid_cap.isOpened():
    ret, frame = vid_cap.read()
    if ret:
        if frame_id >= max_record:
            assert frame.shape == background.shape
            
            bg_id = 1000# ((frame_id//max_record) * max_record)
            background = cv2.imread(f"backgrounds/{vid}_bg/{vid}_background_{bg_id}.jpg")

            inputs = FGD.preprocess(frame, background).cuda()
           
            infer_out = FGD.infer(inputs)
            
            hmap = FGD.update_staticness_map(infer_out)

            fname = f'infer_results/{vid}/infer_batch.jpg' 
            plot_batch(imgs=inputs, masks=infer_out.detach(), fname=fname, save = True, hmap = hmap)
            fr = cv2.imread(fname)
            out.write(fr)
            hmap_out.write(hmap)

            frame_id +=1
            print(frame_id)
        else: 
            frame_id+=1

    else: 
        break
vid_cap.release()
cv2.destroyAllWindows()
out.release()
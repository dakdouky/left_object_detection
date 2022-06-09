import os
import cv2
import numpy as np
from modules.foreground_detector import ForegroundDetector
from utils.utils import * 

import onnxruntime

from modules.background_modeling import BackgroundUpdate
from utils.visualize import plot_batch
import datetime

dev = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = dev
print("Using GPU: ", dev) 

time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 
# load video
video_name = "PETS2006"
video_ext = "mp4"
tracker_file = "2022_05_17_13_06_44.txt"

model_format = "ONNX" # ONNX

# PETS2006.mp4  2022_05_17_13_06_44
# Easy 2022_05_09_15_06_57
# easy_medium  2022_05_16_15_39_28
cap, vid_length, width, height, fps = load_video(video_name, video_ext)

max_record = 1000
dim = (320, 256) # w, h have to be multiple of 32 (grid cell size)
orig_dim = (width, height)

if not os.path.exists('backgrounds/{}_bg/'.format(video_name)):
    os.makedirs('backgrounds/{}_bg/'.format(video_name))

if not os.path.exists(f"infer_results/{video_name}/"):
    os.mkdir(f"infer_results/{video_name}/")
if not os.path.exists(f"infer_results/{video_name}_masks/"):
    os.mkdir(f"infer_results/{video_name}_masks/")
vid_path = f"videos/{video_name}.{video_ext}"
out_path = f"videos/{video_name}_output.mp4"

# read tracker person bounding boxes
#tracker_file = "2022_05_09_15_06_57.txt"
data = load_tracker_data(tracker_file)
detector_data = per_frame_bbox(data, vid_length)

# background modeling
BGM = BackgroundUpdate(dim, max_record)

# foreground detector
model_name = 'saved_models/model_epoch_FCS.mdl'
cuda = True

FGD = ForegitgroundDetector(dim, cuda, model_format)
FGD.laod_model(model_name)


# begin processing
print(f"Processing video: {video_name} ...")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(f'infer_results/output/{time_now}_{video_name}.mp4', fourcc, 8.0, (4000,4000))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#
# hmap_out = cv2.VideoWriter(f'infer_results/output/{time_now}_{video_name}_hmap.mp4', fourcc, 20.0, (320, 240))


frame_id = 0
foreground_boxes = []

while True:
    ret, frame = cap.read()
    if ret:
        
        detector_bboxes = detector_data[str(frame_id)]
        detector_bboxes = scale_bbox(detector_bboxes, orig_dim, dim) # to fit new image dim
        detector_bboxes = enlarge_bbox(detector_bboxes, scale_factor=1.5) # to enclude parts of the object outside the detector bbox

        frame = cv2.resize(frame, (dim[0], dim[1]), interpolation = cv2.INTER_AREA)
        bbox_img = plot_bbox(frame, detector_bboxes)
        print(frame_id)
        BGM.upadate_grid(detector_bboxes, foreground_boxes, frame_id, frame)

        if frame_id > max_record:

            inputs = FGD.preprocess(frame, BGM.background).cuda()
            
            if model_format == "ONNX":
                infer_out = FGD.infer_onnx(inputs)
            else:
                infer_out = FGD.infer(inputs)

            foreground_boxes = FGD.foreground_blobs()
            foreground_boxes = enlarge_bbox(foreground_boxes, scale_factor=1.5) # to enclude parts of the object outside the detector bbox
            
            hmap = FGD.update_staticness_map(detector_bboxes)

            output = FGD.pre_visualize(inputs, infer_out, 0)

            output["image"] = FGD.fire_alarm(output["image"], frame_id, detector_bboxes,
                                             time_threshold = 200, area_threshold = 100)

            fname = f'infer_results/{video_name}/infer_batch.jpg' 
            plot_batch(output = output, bbox_img = bbox_img, fname = fname, save = True, hmap = hmap)
            
            fr = cv2.imread(fname)
            out.write(fr)
            #hmap_out.write(hmap)
 

    frame_id+=1
    if frame_id> vid_length-195:
        break

    if frame_id % max_record == 0: 
        BGM.update_background(frame_id, video_name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


background = BGM.update_background((frame_id//max_record)*max_record + max_record, video_name)

cap.release()
# cv2.destroyAllWindows()

import colorsys
import copy
import random
import cv2 
import numpy as np

# detect if a grid cell is overlapping with a bounding box 
def is_overlapping(box1,box2):
    xmin1, ymin1,xmax1,ymax1 = box1
    xmin2, ymin2,xmax2,ymax2 = box2
   
    return ((xmax1 >= xmin2 and xmax2 >= xmin1) and
                    (ymax1 >= ymin2 and ymax2 >= ymin1))


def load_tracker_data(tracker_file):
    tracker_output_dir = "YOLOX_outputs/yolox_x_mix_det/track_vis/{}".format(tracker_file)
    with open(tracker_output_dir, 'r') as file:
        data = file.readlines()
    return data

def per_frame_bbox(data, vid_length):
    # dictionary mapping frame id to the frame bbox
    bbox = {str(i):[] for i in range(int(vid_length))}
    frame_num = 0
    for b in data:
        if frame_num ==0:
            b = b.split(',')
            frame = b[0]
            box = [int(float((e))) for e in b[2:-4]] 
            bbox[frame].append(box)
    return bbox



def load_video(video_name, video_ext):
    cap = cv2.VideoCapture("videos/{}.{}".format(video_name, video_ext))
    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("vid_length: ", vid_length)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, vid_length, width, height, fps


def plot_bbox(img, bboxes):
    bbox_img = copy.deepcopy(img)
    for box in bboxes:
        x1, y1, w, h = box
        bbox_img = cv2.rectangle(bbox_img, (x1, y1), (x1+w,y1+h), (255, 0, 0), 2)
    
    return bbox_img


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[c, :, :] = np.where(mask == 1,
                                  image[c, :, :] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[c, :, :])
    return image.astype(np.uint8)



def scale_bbox(boxes, original_dim, dim):
    
    scaled_boxes = []     
    w0, h0 = original_dim
    w, h   = dim

    x_scale = w/w0
    y_scale = h/h0
    
    for box in boxes:
        x, y, w, h = box 
        x = int(x * x_scale);
        y = int(y * y_scale);
        w = int(w * x_scale);
        h = int(h * y_scale); 
        scaled_boxes.append([x,y,w,h])
    return scaled_boxes


def enlarge_bbox(boxes, scale_factor = 2):
    """
    enlarge the bbox a little
    """
    enlarged_boxes = []
    for box in boxes:
        x, y, w, h = box 
        
        hc = int(y + h/2)
        wc = int(x + w/2)

        h_new = int(h * scale_factor)
        w_new = int(w * scale_factor)

        x = int(max(wc - w_new/2, 0))
        y = int(max(hc - h_new/2, 0))
        enlarged_boxes.append([x, y, w_new, h_new])

    return enlarged_boxes

def box_area(box):
    _, _, w, h = box
    return w*h

def get_distance(box1, box2):
    x1, y1 = box1
    x2, y2 = box2

    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2) 

    return distance
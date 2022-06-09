from collections import deque
import copy

from cv2 import CV_32FC1
from pandas import Int8Dtype
from utils.utils import apply_mask, box_area, get_distance, is_overlapping, random_colors
import cv2 
import numpy as np
import torchvision.transforms as tvtf
import torch 
import onnxruntime

mean_rgb = [x for x in [0.485, 0.456, 0.406]]
std_rgb = [x for x in [0.229, 0.224, 0.225]]

class ForegroundDetector:

  def __init__(self, dim, cuda, model_format):
    self.cuda = cuda    
    self.staticness_map = np.zeros(dim[::-1], np.float32)
    self.dim = dim
    self.infer_out = None
    self.fg_map = None
    self.model_format = model_format

    
  def laod_model(self, model_name):
    print("Loading model...")
    self.model = torch.load(model_name)
    if self.cuda: self.model.cuda() 
  
  def infer(self, inputs):
    print("Using TORCH model...")
    print(inputs.type)
    self.model.eval()
    self.infer_out = self.model(inputs)
    self.infer_out[self.infer_out>=0.5] = 1.
    self.infer_out[self.infer_out<0.5] = 0.

    return self.infer_out
  
  def infer_onnx(self, inputs):
    print("Using ONNX model...")
    ort_session = onnxruntime.InferenceSession("change_detection.onnx", 
                                        providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    infer_out = ort_session.run(None, ort_inputs)
    self.infer_out = infer_out[0]
    self.infer_out[self.infer_out>=0.5] = 1.
    self.infer_out[self.infer_out<0.5] = 0.
    return self.infer_out


  def update_staticness_map(self, frame_bboxes):
    
    for c,b in enumerate(frame_bboxes):
      x1, y1, w, h = b
      x2, y2 = x1 + w, y1 + h
      self.fg_map[y1:y2, x1:x2] = np.ones_like(self.fg_map[y1:y2, x1:x2]) * -255

    self.staticness_map = np.where(self.fg_map == 255., self.staticness_map + 0.5, 
                          np.where(self.fg_map == 0.,   self.staticness_map - 1.0, self.staticness_map))
    
    self.staticness_map[self.staticness_map<0.] = 0.
    self.staticness_map[self.staticness_map>255.] = 255.


    hmap = cv2.applyColorMap(self.staticness_map.astype(np.uint8), cv2.COLORMAP_JET).astype(np.uint8)

    return hmap

  def preprocess(self, frame, background):
        
    fr = copy.deepcopy(frame)
    bg = copy.deepcopy(background)

    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)/255.
    fr = cv2.resize(fr, self.dim, interpolation= cv2.INTER_AREA)

    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)/255.
    bg = cv2.resize(bg, self.dim, interpolation= cv2.INTER_AREA)

    inp_tensors = []
    inp_tensors.append(tvtf.ToTensor()(bg))
    inp_tensors.append(tvtf.ToTensor()(fr))
    inp_tensor = torch.cat(inp_tensors, dim=0)

    mean_period = mean_rgb.copy()
    std_period = std_rgb.copy()

    num_frames = 2

    mean_vec = np.concatenate([mean_period for _ in range(num_frames)])
    std_vec = np.concatenate([std_period for _ in range(num_frames)])
    inputs = tvtf.Normalize(mean_vec, std_vec)(inp_tensor).float()

    return inputs.unsqueeze(0)

  
  def pre_visualize(self, inputs, masks,  batch):
      
      img = inputs[batch,3:6,:,:].cpu().numpy()
      background = inputs[batch,0:3,:,:].cpu().numpy()

      mean_rgb=[0.485, 0.456, 0.406]
      std_rgb=[0.229, 0.224, 0.225]
      
      if self.model_format == "ONNX":
         masks = masks#.detach().cpu().numpy()
      else: 
         masks = masks.detach().cpu().numpy()

      color = random_colors(1)[0]

      img = img.transpose(1, 2, 0) * std_rgb + mean_rgb
      img = img.transpose(2, 0, 1) * 255.
      mask = masks[batch,:,:]

      masked = apply_mask(img, mask, color)

      background = background.transpose(1, 2, 0) * std_rgb + mean_rgb
      background = background.transpose(2, 0, 1) * 255.

      output = {"image"      :  np.ascontiguousarray(img.transpose(1, 2, 0), dtype=np.uint8), 
                "background" :  np.ascontiguousarray(background.transpose(1, 2, 0), dtype=np.uint8),
                "masked"     :  np.ascontiguousarray(masked, dtype=np.uint8)}

      return output

        

  def foreground_blobs(self):
        
    if self.model_format == "ONNX":
       self.fg_map = self.infer_out[0, 0, :, :].astype(np.uint8) * 255
    else: 
       self.fg_map = self.infer_out.cpu().detach().numpy()[0, 0, :, :].astype(np.uint8) * 255
    #self.fg_map = cv2.resize(self.fg_map, (width, height), interpolation = cv2.INTER_NEAREST)

    mapRGB = cv2.cvtColor(self.fg_map, cv2.COLOR_GRAY2RGB)
    ctrs, _ = cv2.findContours(self.fg_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for ctr in ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        boxes.append([x, y, w, h])

    for box in boxes:
        top_left     = (box[0], box[1])
        bottom_right = (box[0] + box[2], box[1] + box[3])
        cv2.rectangle(mapRGB, top_left, bottom_right, (0,255,0), 2)

    cv2.imwrite("mp.jpg", mapRGB)

    return boxes


  def fire_alarm(self, frame, id, detector_bboxes, time_threshold=50, area_threshold = 100):

    _ ,thresh = cv2.threshold(self.staticness_map, time_threshold, 255, 0)
    thresh = thresh.astype(np.uint8)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for ctr in ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        boxes.append([x, y, w, h])

    for box in boxes:
        if box_area(box) > area_threshold:
          if self.is_attended(box, detector_bboxes):
            color =  (0,255,0)
            print("attended")
          else: 
            color =  (255,0,0)
            print("not attended")


          top_left     = (box[0], box[1])
          bottom_right = (box[0] + box[2], box[1] + box[3])
          cv2.rectangle(frame, top_left, bottom_right, color, 2)

    cv2.imwrite(f"ctr/Contour_{id}.jpg",frame[:,:,::-1])
    return frame


  def is_attended(self, box, detector_bboxes):
    """decides if a detected left objcet has a person nearby
       within a circle of radius: radius"""
    x, y, w, h = box
    xc = int(x + w/2)
    yc = int(y + h/2)
    distance = np.inf
    for b in detector_bboxes:
        x, y, w, h = b
        xc_detect = int(x + w/2)
        yc_detect = int(y + h/2)

        distance = get_distance((xc, yc), (xc_detect, yc_detect))
        
    return True if distance < 70 else False


    
    


    

    
           
    

      


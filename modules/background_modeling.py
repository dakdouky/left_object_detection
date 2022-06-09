from collections import deque
from utils.utils import is_overlapping
import cv2 
import numpy as np

class BackgroundUpdate:
  def __init__(self, dim, max_record):
    self.dim  = dim# w, h
    self.width = dim[0]
    self.height = dim[1]

    self.max_record = max_record
    self.background = None    
    self.grid = self.create_grid()

  def upadate_grid(self, detector_bboxes, foreground_bboxes, frame_id, frame):
    for k in self.grid.keys():
        cell =self.grid[k]['box']

        no_intersection_flag = -1
        
        frame_bboxes = detector_bboxes + foreground_bboxes
 
        for c,b in enumerate(frame_bboxes):
            x1, y1, w, h = b
            x2, y2 = x1 + w, y1 + h
            x1 = max(0, x1)
            y1 = max(0, y1)
            intbox = tuple(map(int, (x1, y1, x2, y2)))

            if is_overlapping(cell,intbox) == 0:
                no_intersection_flag +=1
        if len(frame_bboxes) != 0:
            if no_intersection_flag == c :
                segment = frame[cell[1]:cell[3], cell[0]:cell[2]]
                self.grid[k]['segments'].appendleft(segment) 
    return 

  def update_background(self, frame_id, video_name):
      
      for k in self.grid.keys():
          self.grid[k]['filtered']  = np.median(self.grid[k]['segments'], axis=0).astype(dtype=np.uint8)
          
      background = np.zeros([self.height, self.width, 3])
      for k in self.grid.keys():
          window = self.grid[k]['filtered']
          box = self.grid[k]['box']
          x1,y1,x2,y2 = box
          background[y1:y2, x1:x2] = window
      cv2.imwrite('backgrounds/{}_bg/{}_background_{}.jpg'.format(video_name, video_name, frame_id), background)
      print(" === saved background to: {}_bg/{}_background_{}.jpg ===".format(video_name, video_name, frame_id))
      self.background = np.float32(background)

  def create_grid(self):
    # creating a grid
    M = 32
    N = 32

    dh = int(self.height/M)
    dw = int(self.width/N)
    tiles = [[x,y, x+dw, y+dh] for x in range(0,self.width,dw) for y in range(0,self.height,dh)]
    grid = {i:{'box':tiles[i], 'segments':deque(maxlen=self.max_record) } for i in range(M*N)}
    
    return grid



"""
Visualizations for input and output tensors
"""

import colorsys
import copy
import random
import os
import cv2
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path


def tensor2double(inp, segmentation_ch=False,
              mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225],
              mean_seg=[0.5], std_seg=[0.5]):
    """
    Convert the tensor image into numpy array in the range 0,1
    Assumes batch_size=1

    inp (Tensor of size (CxWxH)): Input tensor of size 1xCxWxH

    returns:
    (np array iof size (WxHXC)): output array
    """

    inp_numpy = inp.cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
    w, h, c = inp_numpy.shape

    num_cahnnels_per_fr = 3+(1*segmentation_ch)
    num_frames = int(c/num_cahnnels_per_fr) # empty bg (?), recent bg (?), current fr


    for ch in range(num_frames):
        im = inp_numpy[:, :, num_cahnnels_per_fr*ch+(1*segmentation_ch):num_cahnnels_per_fr*(ch+1)]
        im = (im*std_rgb)+mean_rgb
        inp_numpy[:, :, num_cahnnels_per_fr*ch+(1*segmentation_ch):num_cahnnels_per_fr*(ch+1)] = im
        if segmentation_ch:
            im = inp_numpy[:, :, num_cahnnels_per_fr*ch]
            im = (im*std_seg) + std_seg
            inp_numpy[:, :, num_cahnnels_per_fr*ch] = im

    return inp_numpy

def visualize(inp, out, segmentation_ch=False,
              mean_rgb=[0.485, 0.456, 0.406], std_rgb=[0.229, 0.224, 0.225],
              mean_seg=[0.5], std_seg=[0.5]):
    """
    Shows the first input and output data from the minibatch in matplotlib

    inp (Tensor of size (CxWxH)): Input tensor of size BxCxWxH
    out (Tensor of size (1xWxH)): Output tensor of size Bx1xWxH
    """

    inp_numpy = inp.cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
    out_numpy = out.cpu().numpy()[0, 0, :, :]
    w, h, c = inp_numpy.shape

    num_cahnnels_per_fr = 3+(1*segmentation_ch)
    num_frames = int(c/num_cahnnels_per_fr) # empty bg (?), recent bg (?), current fr

    fig, axes = plt.subplots((1+(1*segmentation_ch)), num_frames+1, figsize=(40, 20))

    im_arr = []
    for ch in range(num_frames):
        im = inp_numpy[:, :, num_cahnnels_per_fr*ch+(1*segmentation_ch):num_cahnnels_per_fr*(ch+1)]
        im = (im*std_rgb)+mean_rgb
        im_arr.append(im)
        axes[0, ch].imshow(im)
        if segmentation_ch:
            im = inp_numpy[:, :, num_cahnnels_per_fr*ch]
            im = (im*std_seg) + std_seg
            axes[1, ch].imshow(im)

    axes[0, num_frames].imshow(out_numpy)
    im_arr.append(out_numpy)
    return im_arr



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


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
            print(mask[i, :, :].sum())
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()



def plot_batch(output, bbox_img, paths=None, fname='images.jpg', save = False,  hmap = None):
    # Plots training images overlaid with masks
    img = output["image"]
    background =  output["background"]
    masked = output["masked"]

    fig = plt.figure(figsize=(20, 20))

    bs = 2
    ns = np.ceil(bs ** 0.5)  # number of subplots
 
    plt.subplot(2, ns, 1).imshow(img)
    plt.subplot(2, ns, 2).imshow(background)
    plt.subplot(2, ns, 3).imshow(hmap[:, :, ::-1])
    plt.subplot(2, ns, 4).imshow(bbox_img[:, :, ::-1])

    plt.axis('off')

    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()





"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from typing import final
import torch
import os
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorize_depth(x, cmap):
    depth = np.squeeze(x)
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]
    return depth.astype('uint8')

# This function takes a 4D tensor in the form NxCxWxH and save it to images according to idxs
def saveTensorToImage(outputs, vars, labels, inputs_d, inputs_rgb, save_to_path, epoch):
    if os.path.exists(save_to_path) == False:
        os.mkdir(save_to_path)

    cmap_depth = plt.cm.jet
    cmap_uncertainty = plt.cm.viridis
    for t in range(outputs.shape[0]):
        final_img = []
        final_img.append(np.transpose(255 * np.squeeze(inputs_rgb[t]), (1, 2, 0)).astype('uint8'))
        final_img.append(colorize_depth(inputs_d[t], cmap_depth))
        final_img.append(colorize_depth(outputs[t], cmap_depth))
        final_img.append(colorize_depth(vars[t], cmap_uncertainty))
        final_img.append(colorize_depth(labels[t], cmap_depth))
        final_img = np.hstack(final_img).astype('uint8')
        imout = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_to_path, 'epoch_{}_image_{}.png'.format(str(epoch - 1).zfill(4), str(t).zfill(4))), imout)



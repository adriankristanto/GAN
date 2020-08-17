import cv2
import numpy as np
import os
import re

# reference: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
FPS = 10
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_FILENAME = 'dcgan_training.mp4'


filenames = sorted([GENERATED_DIRPATH + filename for filename in os.listdir(GENERATED_DIRPATH) if 'dcgan_sample_' in filename], key=lambda x: int(re.search(r'dcgan\_sample\_(\d+)\.png', x)[1]))
img_array = []
size = None

for i, filename in enumerate(filenames):
    img = cv2.imread(filename)
    height, width, channels = img.shape
    if i == 0:
        size = (height, width)
    img_array.append(img)

out = cv2.VideoWriter(VIDEO_FILENAME, FOURCC, FPS, size)

for img in img_array:
    out.write(img)

out.release()
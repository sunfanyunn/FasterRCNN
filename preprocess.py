import tensorflow as tf
from matplotlib import pyplot as plt
from tensorpack.utils.rect import FloatBox
import numpy as np
import os, sys
import glob
import cv2
import pickle
import json
from PIL import Image
from tqdm import tqdm
from PIL import Image

def get_line(start, end):
    """
    Bresenham's Line Algorithm
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def drawBox(im, box):
    xmin, ymin, xmax, ymax = box
    ymin = int(ymin*im.shape[0])
    xmin = int(xmin*im.shape[0])
    ymax = int(ymax*im.shape[0])
    xmax = int(xmax*im.shape[0])
    top = get_line((ymin, xmin), (ymin, xmax))
    left = get_line((ymin, xmin), (ymax, xmin))
    right = get_line((ymin, xmax), (ymax, xmax))
    bottom = get_line((ymax, xmin), (ymax, xmax))
    for _l in [top, left, right, bottom]:
        for p in _l:
            im[p[1], p[0]] = 2
    return im

def showLabel(image):
    image_ids = set(image[image>0])
    colors = {}
    for _id in image_ids:
        colors[_id] = [int(np.random.random()*255), int(np.random.random()*255), int(np.random.random()*255)]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    for l in image_ids:
        r[image==l] = colors[l][0]
        g[image==l] = colors[l][1]
        b[image==l] = colors[l][2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    return np.uint8(rgb)

# crop (and fusion) then save image
def preprocess(filenames, fusion=True):
    for index, filename in tqdm(enumerate(filenames)):
        basename = os.path.basename(filename)
        basename, _ = os.path.splitext(basename)
        patid, idx = basename.split('_')
        patid, idx = int(patid), int(idx)

        with open(filename, 'rb') as f:
            _file = pickle.load(f)

        if fusion:
            mri_image = plt.imread(mri_path + basename + '.jpg')
            mri_image = cv2.resize(mri_image, (250,250), interpolation=cv2.INTER_LINEAR)
            mri_image = mri_image[25:225, 25:225]

        image_data = _file['image']

        xoffset = (image_data.shape[0]-200)/2
        yoffset = (image_data.shape[1]-200)/2
        assert xoffset == yoffset
        offset = xoffset

        image_data = image_data[xoffset:xoffset+200,yoffset:yoffset+200]
        if fusion:
            image_data[...,1] = mri_image

        assert image_data.shape == (200,200,3)
        image_data = image_data.astype(np.uint8)

        if fusion:
            plt.imsave(os.path.join('/tmp2/oar/ct_segmentation_data_fine/fusion',basename) + '.png', image_data)
        else:
            plt.imsave(os.path.join('/tmp2/oar/ct_segmentation_data_fine/processed_image',basename) + '.png', image_data)

if __name__ == '__main__':
    mri_path = '/tmp2/oar/ct_segmentation_data_fine/mri/'
    filenames = glob.glob(config.BASEDIR + '/**/**/*.pkl')
#    filenames = glob.glob("/tmp2/oar/ct_segmentation_data_fine/test/**/*.pkl") + glob.glob("/tmp2/oar/ct_segmentation_data_fine/train/**/*.pkl") 
    print(len(filenames))
    preprocess(filenames, fusion=False)

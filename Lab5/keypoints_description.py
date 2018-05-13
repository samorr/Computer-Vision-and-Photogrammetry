import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndim
import imageio
import matplotlib.pyplot as plt
import time
import skimage.feature
from scipy import io

data = io.loadmat('data/Episcopal_Gaudi/f_o.mat')
points1 = np.concatenate([data['y1'], data['x1']], axis=1)
points2 = np.concatenate([data['y2'], data['x2']], axis=1)

image = np.asarray(Image.open('data/Episcopal_Gaudi/1_o.jpg', 'r').convert('L')).astype(np.float)

def compute_orientations_histogram(image, point, gradient, coeff):
    point = point.astype(np.int)
    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0
    if point[0] < 20:
        point[0] += 20
        top_pad = 20
    top = point[0] - 20
    bottom = point[0] + 21

    if point[1] < 20:
        point[1] += 20
        left_pad = 20
    left = point[1] - 20
    right = point[1] + 21

    if bottom >= image.shape[0]:
        bottom_pad = 20
    if right >= image.shape[1]:
        right_pad = 20

    grad = np.pad(gradient, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), 'reflect')[top:bottom, left:right]

    ax = np.arange(-41 // 2 + 1., 41 // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * coeff**2))
    kernel /= np.sum(kernel)

    magnitude = np.sqrt(np.sum(grad ** 2, axis=-1)) * kernel
    orientation = (36. * (np.arctan2(grad[:,:,1], grad[:,:,0]) + np.pi) / (2. * np.pi)).astype(np.int)
    orientation[orientation == 36.] = 0.
    histogram = np.array([np.sum(magnitude[orientation == i]) for i in range(36)])

    return histogram

def compute_orientations_histogram_for_all_points(image, points, sigma=1.): # for now I don't know how to vectorized this operation
    coeff = 1.5 * sigma
    gradient = np.stack(np.gradient(image), axis=-1)
    histograms = [compute_orientations_histogram(image, point, gradient, coeff) for point in points]

    return histograms


t = time.time()
compute_orientations_histogram_for_all_points(image, points1)

print('Time for {} points:'.format(points1.shape[0]), time.time() - t)
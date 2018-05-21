import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndim
import imageio
import matplotlib.pyplot as plt
import time
import skimage.feature
from scipy import io

data = io.loadmat('data/Notre_Dame/f_o.mat')
points1 = np.concatenate([data['y1'], data['x1']], axis=1)
points2 = np.concatenate([data['y2'], data['x2']], axis=1)

image1 = np.asarray(Image.open('data/Notre_Dame/1_o.jpg', 'r').convert('L')).astype(np.float)
image2 = np.asarray(Image.open('data/Notre_Dame/2_o.jpg', 'r').convert('L')).astype(np.float)


def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def compute_dominant_orientation(point, gradient, coeff):
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

    if bottom >= gradient.shape[0]:
        bottom_pad = 20
    if right >= gradient.shape[1]:
        right_pad = 20

    grad = np.pad(gradient, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), 'reflect')[top:bottom, left:right]

    kernel = gaussian_kernel(41, coeff)
    magnitude = np.sqrt(np.sum(grad ** 2, axis=-1)) * kernel
    orientation = np.arctan2(grad[:,:,0], grad[:,:,1])
    orientation[orientation < 0.] += 2 * np.pi
    orientation = (36. * orientation / (2. * np.pi)).astype(np.int)
    orientation[orientation == 36.] = 0.
    histogram = np.array([np.sum(magnitude[orientation == i]) for i in range(36)])
    dominant_orientation = np.argmax(histogram)

    return dominant_orientation, grad # later maybe return also orientations which has values near to maximum

def compute_dominant_orientation_for_all_points(points, image, sigma=1.): # for now I don't know how to vectorized this operation
    coeff = 1.5 * sigma
    gradient = np.stack(np.gradient(image), axis=-1)
    dominant_orientations = [compute_dominant_orientation(point, gradient, coeff) for point in points]

    return dominant_orientations

def compute_descriptor(point, image_gradient, sigma=1.):
    coeff = 1.5 * sigma
    dominant_orientation, big_patch_gradient = compute_dominant_orientation(point, image_gradient, coeff)
    center = 20
    patch_gradient = ndim.interpolation.rotate(big_patch_gradient, dominant_orientation * 10, reshape=False, prefilter=False)[center-8 : center+8, center-8 : center+8].reshape((16*16,2))
    kernel = gaussian_kernel(16, 4.).reshape(16*16)
    magnitude = np.sqrt(np.sum(patch_gradient ** 2, axis=-1)) * kernel
    orientation = np.arctan2(patch_gradient[:,0], patch_gradient[:,1])
    orientation[orientation < 0.] += 2 * np.pi
    orientation = 8. * orientation / (2 * np.pi)
    orientation[orientation == 8.] = 0.
    
    descriptor = np.empty((16,8))
    floor = np.floor(orientation).astype(np.int)
    ceil = np.ceil(orientation).astype(np.int) # ceil and floor are arrays of histogram bins indices to which gradient vector contributes
    ceil[ceil == 8] = 0
    base = np.array([[0., 1.], [1., 1.], [1., 0.], [1., -1.], [0., -1.], [-1., -1.], [-1., 0.], [-1., 1.]])
    base /= np.sqrt(np.sum(base ** 2, axis=1))[:, np.newaxis] # these are vectors that represents histogram bins
    base_per_vector = np.stack((base[ceil], base[floor]), axis=-1)
    base_per_vector[ceil == floor, 0, 1] = base_per_vector[ceil == floor, 1, 0]
    base_per_vector[ceil == floor, 1, 1] = -base_per_vector[ceil == floor, 0, 0] # if gradient is collinear with base vector then second one is set to be orthogonal
    coeffs = np.linalg.solve(base_per_vector, patch_gradient) # coefficients of gradient magnitude contribution in appropriate histogram bins
    coeffs_sum = np.sum(coeffs, axis=1)
    coeffs_sum[coeffs_sum == 0.] = 1.
    coeffs /= coeffs_sum[:, np.newaxis]
    
    local_desc = np.zeros((256,8))
    ind = np.arange(256)
    local_desc[ind, ceil] = magnitude * coeffs[:,0]
    local_desc[ind, floor] = magnitude * coeffs[:,1]
    
    local_desc = local_desc.reshape((64,4,8))
    for i in range(4):
        descriptor[4*i : 4*(i+1)] = np.sum(local_desc[16*i:16*(i+1):4,:,:],axis=0) # meybe vectorize

    descriptor = descriptor.reshape(128)
    descriptor /= np.sqrt(np.sum(descriptor ** 2))
    descriptor[descriptor > 0.2] = 0.2
    descriptor /= np.sqrt(np.sum(descriptor ** 2))

    return descriptor

def compute_descriptors_for_all_points(points, image):
    gradient = np.stack(np.gradient(image), axis=-1)
    descriptors = np.array([compute_descriptor(point, gradient) for point in points])
    return descriptors


def match_keypoints(descriptors_image1, descriptors_image2):
    distances = np.sqrt(np.sum((descriptors_image1[:, np.newaxis, :] - descriptors_image2[np.newaxis, :, :]) ** 2, axis=-1))
    best_matches_indices = np.argpartition(distances,2, axis=1)[:,:2]
    ind = np.arange(descriptors_image1.shape[0])[:,np.newaxis]
    matches_distances = distances[ind, best_matches_indices]
    ratio = matches_distances[:,0] / matches_distances[:,1]
    sorted_indices = np.argsort(ratio)
    return best_matches_indices[sorted_indices, :], matches_distances[sorted_indices, :], ratio[sorted_indices], ind[sorted_indices]


# t = time.time()
# compute_dominant_orientation_for_all_points(points1, image)

# t = time.time()
# compute_descriptors_for_all_points(points1, image)

# print('Time for {} points:'.format(points1.shape[0]), time.time() - t)

desc1 = compute_descriptors_for_all_points(points1, image1)
desc2 = compute_descriptors_for_all_points(points2, image2)
matches, matches_dists, ratio, ind = match_keypoints(desc1, desc2)
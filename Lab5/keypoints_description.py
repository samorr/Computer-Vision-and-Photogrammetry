import sys
sys.path.append('..')
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndim
import matplotlib.pyplot as plt
import time
from scipy import io
from Lab4 import keypoints_detection as detect

# Mount_Rushmore
# Episcopal_Gaudi
# Notre_Dame

def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel

def compute_dominant_orientation(point_l, gradient, coeff):
    point = point_l.astype(np.int)
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
    magnitude = np.sqrt(np.sum(grad ** 2., axis=-1)) * kernel
    orientation = np.arctan2(grad[:,:,0], grad[:,:,1])
    orientation[orientation < 0.] += 2. * np.pi
    orientation = (36. * orientation / (2. * np.pi)).astype(np.int)
    orientation[orientation == 36.] = 0.
    histogram = np.array([np.sum(magnitude[orientation == i]) for i in range(36)])
    dominant_orientation = np.argmax(histogram)

    return dominant_orientation, grad # later maybe return also orientations which has values near to maximum

def compute_dominant_orientation_for_all_points(points, image, sigma=1.):
    coeff = 1.5 * sigma
    gradient = np.stack(np.gradient(image), axis=-1)
    dominant_orientations = [compute_dominant_orientation(point, gradient, coeff) for point in points]

    return dominant_orientations

def compute_descriptor(point, image_gradient, sigma=1.):
    DESCRIPTOR_WINDOW_SIZE = 16 # must be divisible by 4
    coeff = 1.5 * sigma
    dominant_orientation, big_patch_gradient = compute_dominant_orientation(point, image_gradient, coeff)
    center = 20
    patch_gradient = ndim.interpolation.rotate(big_patch_gradient, dominant_orientation * 10, reshape=False, prefilter=False)[center - DESCRIPTOR_WINDOW_SIZE // 2 : center + DESCRIPTOR_WINDOW_SIZE // 2, center - DESCRIPTOR_WINDOW_SIZE // 2 : center + DESCRIPTOR_WINDOW_SIZE // 2].reshape((DESCRIPTOR_WINDOW_SIZE * DESCRIPTOR_WINDOW_SIZE, 2))
    kernel = gaussian_kernel(DESCRIPTOR_WINDOW_SIZE, DESCRIPTOR_WINDOW_SIZE / 4.).reshape(DESCRIPTOR_WINDOW_SIZE * DESCRIPTOR_WINDOW_SIZE)
    magnitude = np.sqrt(np.sum(patch_gradient ** 2, axis=-1)) * kernel
    orientation = np.arctan2(patch_gradient[:,0], patch_gradient[:,1])
    orientation[orientation < 0.] += 2. * np.pi
    orientation = 8. * orientation / (2. * np.pi)
    orientation[orientation == 8.] = 0.
    
    descriptor = np.empty((DESCRIPTOR_WINDOW_SIZE, 8))
    floor = np.floor(orientation).astype(np.int)
    ceil = np.ceil(orientation).astype(np.int) # ceil and floor are arrays of histogram bins indices to which gradient vector contributes
    
    coeffs = np.empty((DESCRIPTOR_WINDOW_SIZE * DESCRIPTOR_WINDOW_SIZE, 2)) # coefficients of gradient magnitude contribution in appropriate histogram bins
    mask = floor == ceil
    coeffs[mask, 0] = 1.
    coeffs[mask, 1] = 0.
    coeffs[~mask,0] = (ceil - orientation)[~mask]
    coeffs[~mask,1] = (orientation - floor)[~mask]
    ceil[ceil == 8] = 0
    
    
    local_desc = np.zeros((DESCRIPTOR_WINDOW_SIZE * DESCRIPTOR_WINDOW_SIZE, 8))
    ind = np.arange(DESCRIPTOR_WINDOW_SIZE * DESCRIPTOR_WINDOW_SIZE)
    local_desc[ind, floor] = magnitude * coeffs[:,0]
    local_desc[ind, ceil] = magnitude * coeffs[:,1]
    
    local_desc = local_desc.reshape((DESCRIPTOR_WINDOW_SIZE * 4, 4, 8))
    for i in range(4):
        descriptor[4*i : 4*(i+1)] = np.sum(local_desc[DESCRIPTOR_WINDOW_SIZE * i : DESCRIPTOR_WINDOW_SIZE * (i+1) : 4, :, :],axis=0) # meybe vectorize

    descriptor = descriptor.reshape(DESCRIPTOR_WINDOW_SIZE * 8)
    descriptor /= np.sqrt(np.sum(descriptor ** 2))
    descriptor[descriptor > 0.2] = 0.2
    descriptor /= np.sqrt(np.sum(descriptor ** 2))

    return descriptor

def compute_descriptors_for_all_points(points, image, sigmas=None):
    gradient = np.stack(np.gradient(image), axis=-1)
    if sigmas is None:
        descriptors = np.array([compute_descriptor(point, gradient) for point in points])
    else:
        descriptors = np.array([compute_descriptor(point, gradient, sigma) for point, sigma in zip(points, sigmas)])
    return descriptors


def match_keypoints(descriptors_image1, descriptors_image2):
    distances = np.sqrt(np.sum((descriptors_image1[:, np.newaxis, :] - descriptors_image2[np.newaxis, :, :]) ** 2, axis=-1))
    best_matches_indices = np.argpartition(distances,2, axis=1)[:,:2]
    ind = np.arange(descriptors_image1.shape[0])[:,np.newaxis]
    matches_distances = distances[ind, best_matches_indices]
    ratio = matches_distances[:,0] / matches_distances[:,1]
    sorted_indices = np.argsort(ratio)
    return np.stack([ind[sorted_indices,0], best_matches_indices[sorted_indices, 0]], axis=1), matches_distances[sorted_indices, 0], ratio[sorted_indices]

def draw_matching(filename1, filename2, points1, points2, matched_ind, new_filename, file_extension='JPEG'):
    im1 = Image.open(filename1)
    im2 = Image.open(filename2)
    totalWidth = im1.size[0] + im2.size[0]
    maxHeight = max(im1.size[1], im2.size[1])
    imOut = Image.new('RGB', (totalWidth, maxHeight))
    offset = im1.size[0]
    imOut.paste(im1, (0, 0))
    imOut.paste(im2, (offset, 0))
    draw = ImageDraw.Draw(imOut)
    width = 2
    for ind in matched_ind:
        if ind[0] == ind[1]:
            fill = (255,0,0)
        else:
            fill = (0,255,0)
        x1 = points1[ind[0],1]
        y1 = points1[ind[0],0]
        draw.ellipse((x1 - width, y1 - width, x1 + width, y1 + width), fill=(0, 255, 0))
        draw.point(points1[ind[0]], fill=(0, 255, 0))
        x2 = points2[ind[1],1] + offset
        y2 = points2[ind[1],0]
        draw.ellipse((x2 - width, y2 - width, x2 + width, y2 + width), fill=(0, 255, 0))
        draw.point(points2[ind[1]], fill=(0, 255, 0))
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=2)
    imOut.save(new_filename, file_extension)

def good_matches(matched_ind, ratio, threshold):
    return np.count_nonzero((matched_ind[:,0] == matched_ind[:,1])[ratio < threshold
    ]), np.count_nonzero(ratio < threshold)

def precision(matched_ind, ratio, threshold):
    return np.count_nonzero((matched_ind[:,0] == matched_ind[:,1])[ratio < threshold
    ]) / np.count_nonzero(ratio < threshold)

def recall(matched_ind, ratio, threshold):
    return np.count_nonzero((matched_ind[:,0] == matched_ind[:,1])[ratio < threshold
    ]) / np.count_nonzero(matched_ind[:,0] == matched_ind[:,1])

def remove_duplicates(matches, ratio):
    matched = set()
    for i in range(matches.shape[0]):
        if matches[i,1] in matched:
            ratio[i] = 1.
        else:
            matched.add(matches[i,1])


if __name__ == '__main__':
    data = io.loadmat('data/Notre_Dame/f_o.mat')
    points1 = np.concatenate([data['y1'], data['x1']], axis=1)
    points2 = np.concatenate([data['y2'], data['x2']], axis=1)

    image1 = ndim.gaussian_filter(np.array(Image.open('data/Notre_Dame/1_o.jpg', 'r').convert('L')).astype(np.float), sigma=3.)
    image2 = ndim.gaussian_filter(np.array(Image.open('data/Notre_Dame/2_o.jpg', 'r').convert('L')).astype(np.float), sigma=3.)

    # t = time.time()
    # compute_dominant_orientation_for_all_points(points1, image)

    # t = time.time()
    # compute_descriptors_for_all_points(points1, image)

    # print('Time for {} points:'.format(points1.shape[0]), time.time() - t)

    # points1 = detect.harris_corner_with_ANMS(image1, 150)
    # points2 = detect.harris_corner_with_ANMS(image2, 150)

    desc1 = compute_descriptors_for_all_points(points1, image1)
    desc2 = compute_descriptors_for_all_points(points2, image2)
    matches, matches_dists, ratio = match_keypoints(desc1, desc2)
    remove_duplicates(matches, ratio)
    threshold = 0.8
    well_matched, all_matches = good_matches(matches, ratio, threshold)
    print('Well matched {} of {}'.format(well_matched, all_matches))
    print('Precision:', precision(matches, ratio, threshold))
    print('Recall:', recall(matches, ratio, threshold))


    # draw_matching('data/Notre_Dame/1_o.jpg', 'data/Notre_Dame/2_o.jpg', points1, points2, matches[ratio < 0.7]], 'data/Notre_Dame/my_detecting_and_matching_with_removing.jpg')

    # draw_matching('data/Mount_Rushmore/1_o.jpg', 'data/Mount_Rushmore/2_o.jpg', points1, points2, matches, 'ground_truth_Mount_Rushmore.jpg')

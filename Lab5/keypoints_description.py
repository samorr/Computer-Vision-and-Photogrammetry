import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndim
import matplotlib.pyplot as plt
import time
from scipy import io
import keypoints_detection as detect

# Mount_Rushmore
# Episcopal_Gaudi
# Notre_Dame

data = io.loadmat('data/Notre_Dame/f_o.mat')
points1 = np.concatenate([data['y1'], data['x1']], axis=1)
points2 = np.concatenate([data['y2'], data['x2']], axis=1)

image1 = ndim.gaussian_filter(np.array(Image.open('data/Notre_Dame/1_o.jpg', 'r').convert('L')).astype(np.float), sigma=3.)
image2 = ndim.gaussian_filter(np.array(Image.open('data/Notre_Dame/2_o.jpg', 'r').convert('L')).astype(np.float), sigma=3.)


def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    # kernel /= np.sum(kernel)
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
    # return (patch_gradient * kernel[:,np.newaxis]).reshape(-1)
    # imageio.imsave('temp.jpg', magnitude.reshape((16,16)) / magnitude.max() * 255)
    # print(magnitude.max())
    orientation = np.arctan2(patch_gradient[:,0], patch_gradient[:,1])
    orientation[orientation < 0.] += 2. * np.pi
    orientation = 8. * orientation / (2. * np.pi)
    orientation[orientation == 8.] = 0.
    
    descriptor = np.empty((DESCRIPTOR_WINDOW_SIZE, 8))
    floor = np.floor(orientation).astype(np.int)
    ceil = np.ceil(orientation).astype(np.int) # ceil and floor are arrays of histogram bins indices to which gradient vector contributes
    # base = np.array([[0., 1.], [1., 1.], [1., 0.], [1., -1.], [0., -1.], [-1., -1.], [-1., 0.], [-1., 1.]]) # mayby just interpolation?
    # base /= np.sqrt(np.sum(base ** 2, axis=1))[:, np.newaxis] # these are vectors that represents histogram bins
    # base_per_vector = np.stack((base[ceil], base[floor]), axis=-1)
    # base_per_vector[ceil == floor, 0, 1] = base_per_vector[ceil == floor, 1, 0]
    # base_per_vector[ceil == floor, 1, 1] = -base_per_vector[ceil == floor, 0, 0] # if gradient is collinear with base vector then second one is set to be orthogonal
    # coeffs = np.linalg.solve(base_per_vector, patch_gradient) # coefficients of gradient magnitude contribution in appropriate histogram bins
    # coeffs_sum = np.sum(coeffs, axis=1)
    # coeffs_sum[coeffs_sum == 0.] = 1.
    # coeffs /= coeffs_sum[:, np.newaxis]
    
    coeffs = np.empty((DESCRIPTOR_WINDOW_SIZE * DESCRIPTOR_WINDOW_SIZE, 2))
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

def match_keypoints_with_removing(descriptors_image1, descriptors_image2, iters_num, threshold=0.7): # this function must be done differently, global indices aren't computed correctly
    best_ind1, _, ratio1, ind1 = match_keypoints(descriptors_image1, descriptors_image2)
    best_ind2, _, ratio2, ind2 = match_keypoints(descriptors_image2, descriptors_image1)
    global_best_ind1 = best_ind1[ratio1 < threshold]
    global_best_ind2 = best_ind2[ratio2 < threshold]
    global_ind1 = ind1[ratio1 < threshold]
    global_ind2 = ind2[ratio2 < threshold]
    for i in range(1, iters_num):
        descriptors_image1 = descriptors_image1[ratio1 < threshold]
        descriptors_image2 = descriptors_image2[ratio2 < threshold]
        best_ind1, best_dists1, ratio1, ind1 = match_keypoints(descriptors_image1, descriptors_image2)
        best_ind2, best_dists2, ratio2, ind2 = match_keypoints(descriptors_image2, descriptors_image1)
        global_ind1 = global_ind1[ratio1 < threshold]
        global_ind2 = global_ind2[ratio2 < threshold]
        global_best_ind1 = global_best_ind1[ratio1 < threshold]
        global_best_ind2 = global_best_ind2[ratio2 < threshold]
    return global_best_ind1, best_dists1, ratio1, global_ind1

def draw_matching(filename1, filename2, points1, points2, matched_ind, new_filename):
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
    imOut.save(new_filename, 'JPEG')

# t = time.time()
# compute_dominant_orientation_for_all_points(points1, image)

# t = time.time()
# compute_descriptors_for_all_points(points1, image)

# print('Time for {} points:'.format(points1.shape[0]), time.time() - t)

points1 = detect.harris_corner_with_ANMS(image1, 150)
points2 = detect.harris_corner_with_ANMS(image2, 150)

desc1 = compute_descriptors_for_all_points(points1, image1)
desc2 = compute_descriptors_for_all_points(points2, image2)
matches, matches_dists, ratio, ind = match_keypoints(desc1, desc2)

# primitive way of duplicates removing
matched = set()
for i in range(matches.shape[0]):
    if matches[i,0] in matched:
        ratio[i] = 1.
    else:
        matched.add(matches[i,0])

draw_matching('data/Notre_Dame/1_o.jpg', 'data/Notre_Dame/2_o.jpg', points1, points2, np.concatenate([ind[ratio < 0.7], matches[:,0,np.newaxis][ratio < 0.7]], axis=1), 'data/Notre_Dame/my_detecting_and_matching_with_removing.jpg')

# draw_matching('data/Mount_Rushmore/1_o.jpg', 'data/Mount_Rushmore/2_o.jpg', points1, points2, np.stack([np.arange(points1.shape[0]), np.arange(points2.shape[0])], axis=1), 'ground_truth_Mount_Rushmore.jpg')

# match_keypoints_with_removing(desc1, desc2, 2, threshold=0.9)

import sys
sys.path.append('..')
from Lab5 import keypoints_description as describe
from Lab4 import keypoints_detection as detect
from Lab3 import epipoles_and_triangulation as reconstruct
from Lab2.lens_distortion import to_homogenous
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
from scipy import io

def ransac_essential_matrix(hom_points1, hom_points2, outliers_threshold, iters=10000):
    max_inliers = 0
    best_sample = np.empty(8, np.int)

    for _ in range(iters):
        sample = np.random.choice(hom_points1.shape[0], 8, replace=False)

        E = reconstruct.essential_matrix(hom_points1[sample,:], hom_points2[sample,:], K)
        F = np.linalg.inv(K.T).dot(E.dot(np.linalg.inv(K)))
        dists = reconstruct.get_dists_from_epipolar_lines(hom_points1, hom_points2, F)
        if np.count_nonzero(dists < outliers_threshold) > max_inliers:
            best_sample = sample
            max_inliers = np.count_nonzero(dists < outliers_threshold)

    E = reconstruct.essential_matrix(hom_points1[best_sample,:], hom_points2[best_sample,:], K)
    F = np.linalg.inv(K.T).dot(E.dot(np.linalg.inv(K)))
    dists = reconstruct.get_dists_from_epipolar_lines(hom_points1, hom_points2, F)

    inliers1 = hom_points1[dists < outliers_threshold,:]
    inliers2 = hom_points2[dists < outliers_threshold,:]
    E = reconstruct.essential_matrix(inliers1, inliers2, K)
    F = np.linalg.inv(K.T).dot(E.dot(np.linalg.inv(K)))
    dists = reconstruct.get_dists_from_epipolar_lines(inliers1, inliers2, F)
    return E, F, dists, inliers1, inliers2


if __name__ == '__main__':

    data_filepath = 'fountain_dense/urd'
    keypoints_filepath = 'fountain_dense/keypoints'
    matching_filepath = 'fountain_dense/matching'
    K = np.array([[2759.48, 0.00000, 1520.69],
                  [0.00000, 2764.16, 1006.81],
                  [0.00000, 0.00000, 1.00000]])

    image1 = np.array(Image.open(data_filepath + '/0000.png', 'r').convert('L')).astype(np.float)
    image2 = np.array(Image.open(data_filepath + '/0001.png', 'r').convert('L')).astype(np.float)

    sigma_step = 1.2
    steps_num = 13
    initial_sigma = 1.5
    keypoints1 = detect.harris_laplace(image1, 7., 2e1, initial_sigma, steps_num, sigma_step)
    detect.draw_points_with_scale_markers(data_filepath + '/0000.png', keypoints1, (255,0,0), keypoints_filepath + '/0000keypoints.png', 1.5, 1.2, file_extension='PNG')

    keypoints2 = detect.harris_laplace(image2, 7., 2e1, initial_sigma, steps_num, sigma_step)
    detect.draw_points_with_scale_markers(data_filepath + '/0001.png', keypoints2, (255,0,0), keypoints_filepath + '/0001keypoints.png', 1.5, 1.2, file_extension='PNG')

    sigmas1 = np.concatenate([np.ones(keypoints1[i].shape[0]) * sigma_step ** i for i in range(len(keypoints1))])
    sigmas2 = np.concatenate([np.ones(keypoints2[i].shape[0]) * sigma_step ** i for i in range(len(keypoints2))])

    keypoints1 = np.concatenate(keypoints1, axis=0)
    keypoints2 = np.concatenate(keypoints2, axis=0)

    desc1 = describe.compute_descriptors_for_all_points_with_scales(keypoints1, image1, sigmas1)
    desc2 = describe.compute_descriptors_for_all_points_with_scales(keypoints2, image2, sigmas2)

    matches, matches_dists, ratio = describe.match_keypoints(desc1, desc2)
    describe.remove_duplicates(matches, ratio)
    threshold = 0.8
    describe.draw_matching(data_filepath + '/0000.png', data_filepath + '/0001.png', keypoints1, keypoints2, matches[ratio < threshold], matching_filepath + '/0000-0001.png', 'PNG')

    matched_points1 = keypoints1[matches[ratio < threshold, 0], :]
    matched_points2 = keypoints2[matches[ratio < threshold, 1], :]

    hom_points1 = to_homogenous(matched_points1)
    hom_points2 = to_homogenous(matched_points2)

    E, F, dists, inliers1, inliers2 = ransac_essential_matrix(hom_points1, hom_points2, 5., iters=100000)

    im = Image.open(data_filepath + '/0000.png') # drawing epipolar lines on image1
    draw = ImageDraw.Draw(im)

    for point in inliers2:
        draw = reconstruct.draw_epipolar_line_yx(point, F, draw, im.size[0])
    width = 3
    fill = (0,255,0)
    for point in inliers1:
        y = point[0]
        x = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save(matching_filepath + '/epipolar-lines-0000.png', 'PNG')

    im = Image.open(data_filepath + '/0001.png') # drawing epipolar lines on image1
    draw = ImageDraw.Draw(im)

    for point in inliers1:
        draw = reconstruct.draw_epipolar_line_yx(point, F.T, draw, im.size[0])
    width = 3
    fill = (0,255,0)
    for point in inliers2:
        y = point[0]
        x = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save(matching_filepath + '/epipolar-lines-0001.png', 'PNG')

    ind = np.stack([np.arange(dists.shape[0]), np.arange(dists.shape[0])], axis=1)
    describe.draw_matching(data_filepath + '/0000.png', data_filepath + '/0001.png', inliers1, inliers2, ind, matching_filepath + '/final-matching-0000-0001.png', 'PNG')
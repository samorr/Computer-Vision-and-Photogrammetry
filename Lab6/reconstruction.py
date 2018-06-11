import sys
sys.path.append('..')
from Lab5 import keypoints_description as describe
from Lab4 import keypoints_detection as detect
from Lab3 import epipoles_and_triangulation as reconstruct
from Lab2.lens_distortion import to_homogenous
import Lab1.projection_matrix as projection
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndim
import matplotlib.pyplot as plt
import time
from scipy import io
import _pickle
import bz2

def bz_pickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    _pickle.dump(obj, f)
    f.close()

def bz_unpickle(filename):
    return _pickle.load(bz2.BZ2File(filename))

def ransac_essential_matrix(hom_points1, hom_points2, outliers_threshold, iters=10000):
    '''Compute essential matrix from image2 to image1 with RANSAC'''

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
    # E = reconstruct.essential_matrix(inliers1, inliers2, K) # maybe stay with E only from best_sample
    # F = np.linalg.inv(K.T).dot(E.dot(np.linalg.inv(K)))
    # dists = reconstruct.get_dists_from_epipolar_lines(inliers1, inliers2, F)
    return E, F, dists, inliers1, inliers2, best_sample


if __name__ == '__main__':

    data_filepath = 'fountain_dense/urd'
    keypoints_filepath = 'fountain_dense/keypoints'
    matching_filepath = 'fountain_dense/matching'
    K = np.array([[2759.48, 0.00000, 1520.69],
                  [0.00000, 2764.16, 1006.81],
                  [0.00000, 0.00000, 1.00000]])

    # reading images
    image1 = np.array(Image.open(data_filepath + '/0000.png', 'r').convert('L')).astype(np.float).T
    image2 = np.array(Image.open(data_filepath + '/0001.png', 'r').convert('L')).astype(np.float).T

    # detecting keypoints
    sigma_step = 1.2
    steps_num = 13
    initial_sigma = 1.5
    keypoints1 = detect.harris_laplace(image1, 7., 2e1, initial_sigma, steps_num, sigma_step)
    detect.draw_points_with_scale_markers(data_filepath + '/0000.png', list(map(np.fliplr, keypoints1)), (255,0,0), \
                                          keypoints_filepath + '/0000keypoints.png', 1.5, 1.2, \
                                          file_extension='PNG')

    keypoints2 = detect.harris_laplace(image2, 7., 2e1, initial_sigma, steps_num, sigma_step)
    detect.draw_points_with_scale_markers(data_filepath + '/0001.png', list(map(np.fliplr, keypoints2)), (255,0,0), \
                                          keypoints_filepath + '/0001keypoints.png', 1.5, 1.2, \
                                          file_extension='PNG')

    # adding scale information to every keypoint
    sigmas1 = np.concatenate([np.ones(keypoints1[i].shape[0]) * sigma_step ** i for i in range(len(keypoints1))])
    sigmas2 = np.concatenate([np.ones(keypoints2[i].shape[0]) * sigma_step ** i for i in range(len(keypoints2))])

    keypoints1 = np.concatenate(keypoints1, axis=0)
    keypoints2 = np.concatenate(keypoints2, axis=0)

    # computing descriptors
    desc1 = describe.compute_descriptors_for_all_points_with_scales(keypoints1, image1, sigmas1)
    desc2 = describe.compute_descriptors_for_all_points_with_scales(keypoints2, image2, sigmas2)

    # finding matches
    matches, matches_dists, ratio = describe.match_keypoints(desc1, desc2)
    describe.remove_duplicates(matches, ratio)
    threshold = 0.8
    describe.draw_matching(data_filepath + '/0000.png', data_filepath + '/0001.png', \
                           np.fliplr(keypoints1), np.fliplr(keypoints2), matches[ratio < threshold], \
                           matching_filepath + '/0000-0001.png', 'PNG')

    matched_points1 = keypoints1[matches[ratio < threshold, 0], :]
    matched_points2 = keypoints2[matches[ratio < threshold, 1], :]
    # desc1 = desc1[matches[:,0], :]
    # desc2 = desc2[matches[:,1], :]

    hom_points1 = to_homogenous(matched_points1)
    hom_points2 = to_homogenous(matched_points2)

    # compute esseintial matrix with RANSAC
    outliers_threshold = 5.
    E, F, dists, inliers1, inliers2, best_sample = ransac_essential_matrix(hom_points1, hom_points2, outliers_threshold, iters=100000)

    im = Image.open(data_filepath + '/0000.png') # drawing epipolar lines on image1
    draw = ImageDraw.Draw(im)

    for point in inliers2:
        draw = reconstruct.draw_epipolar_line(point, F, draw, im.size[0])
    width = 3
    fill = (0,255,0)
    for point in inliers1:
        y = point[1]
        x = point[0]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save(matching_filepath + '/epipolar-lines-0000.png', 'PNG')

    im = Image.open(data_filepath + '/0001.png') # drawing epipolar lines on image2
    draw = ImageDraw.Draw(im)

    for point in inliers1:
        draw = reconstruct.draw_epipolar_line(point, F.T, draw, im.size[0])
    width = 3
    fill = (0,255,0)
    for point in inliers2:
        y = point[1]
        x = point[0]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save(matching_filepath + '/epipolar-lines-0001.png', 'PNG')

    ind = np.stack([np.arange(inliers1.shape[0]), np.arange(inliers1.shape[0])], axis=1)
    describe.draw_matching(data_filepath + '/0000.png', data_filepath + '/0001.png', \
                           np.fliplr(inliers1[:,:-1]), np.fliplr(inliers2[:,:-1]), ind, matching_filepath + '/final-matching-0000-0001.png', 'PNG')
    np.savetxt('fountain_dense/matching/matching_indices-0000-0001.txt', matches[ratio < threshold][dists < outliers_threshold])

    norm_points1 = np.linalg.inv(K).dot(inliers1.T).T
    norm_points2 = np.linalg.inv(K).dot(inliers2.T).T
    P2, P1 = reconstruct.get_projection_matrices_from_essential(E, inliers2, inliers1)
    hom_points_3d = reconstruct.triangulate_points(P2, P1, norm_points2, norm_points1, 'first_3D.ply')
    err1,_ = projection.compute_error(norm_points1, hom_points_3d, P1)
    err2,_ = projection.compute_error(norm_points2, hom_points_3d, P2)



    image3 = np.array(Image.open(data_filepath + '/0002.png', 'r').convert('L')).astype(np.float).T
    keypoints3 = detect.harris_laplace(image3, 7., 2e1, initial_sigma, steps_num, sigma_step)
    detect.draw_points_with_scale_markers(data_filepath + '/0002.png', list(map(np.fliplr, keypoints3)), (255,0,0), \
                                          keypoints_filepath + '/0002keypoints.png', 1.5, 1.2, \
                                          file_extension='PNG')
    sigmas3 = np.concatenate([np.ones(keypoints3[i].shape[0]) * sigma_step ** i for i in range(len(keypoints3))])
    keypoints3 = np.concatenate(keypoints3, axis=0)
    desc3 = describe.compute_descriptors_for_all_points_with_scales(keypoints3, image3, sigmas3)

    desc23_2 = desc2[matches[ratio < threshold, 1], :][dists < outliers_threshold]

    matches23, matches_dists23, ratio23 = describe.match_keypoints(desc23_2, desc3)
    describe.remove_duplicates(matches23, ratio23)
    threshold = 0.8
    describe.draw_matching(data_filepath + '/0001.png', data_filepath + '/0002.png', \
                           np.fliplr(inliers2[:,:-1]), np.fliplr(keypoints3), matches23[ratio23 < threshold], \
                           matching_filepath + '/0001-0002-known3D.png', 'PNG')

    matched_points23_2 = inliers2[matches23[ratio23 < threshold, 0], :]
    matched_points23_3 = keypoints3[matches23[ratio23 < threshold, 1], :]
    hom_points23_2 = matched_points23_2
    hom_points23_3 = to_homogenous(matched_points23_3)
    norm_points23_2 = np.linalg.inv(K).dot(hom_points23_2.T).T
    norm_points23_3 = np.linalg.inv(K).dot(hom_points23_3.T).T

    max_inliers = 0
    best_sample23 = np.empty(6, np.int)
    outliers_threshold = 1e-3

    iters = 10000
    for _ in range(iters):
        sample = np.random.choice(norm_points23_2.shape[0], 6, replace=False)

        P3 = projection.projection_matrix(norm_points23_3[sample], hom_points_3d[matches23[ratio23 < threshold, 0], :][sample])
        err, err_sum = projection.compute_error(norm_points23_3, hom_points_3d[matches23[ratio23 < threshold, 0], :], P3)
        # print(err)
        if np.count_nonzero(err < outliers_threshold) > max_inliers:
            best_sample23 = sample
            max_inliers = np.count_nonzero(err < outliers_threshold)

    P3 = projection.projection_matrix(norm_points23_3[best_sample23], hom_points_3d[matches23[ratio23 < threshold, 0], :][best_sample23])

    matches23, matches_dists23, ratio23 = describe.match_keypoints(desc2, desc3)
    describe.remove_duplicates(matches23, ratio23)
    threshold = 0.8
    describe.draw_matching(data_filepath + '/0001.png', data_filepath + '/0002.png', \
                           np.fliplr(keypoints2), np.fliplr(keypoints3), matches23[ratio23 < threshold], \
                           matching_filepath + '/0001-0002-all.png', 'PNG')

    matched_points23_2 = keypoints2[matches23[ratio23 < threshold, 0], :]
    matched_points23_3 = keypoints3[matches23[ratio23 < threshold, 1], :]
    hom_points23_2 = to_homogenous(matched_points23_2)
    hom_points23_3 = to_homogenous(matched_points23_3)
    norm_points23_2 = np.linalg.inv(K).dot(hom_points23_2.T).T
    norm_points23_3 = np.linalg.inv(K).dot(hom_points23_3.T).T

    err_threshold = 1e-5
    hom_points_3d_23 = reconstruct.triangulate_points(P2, P3, norm_points23_2, norm_points23_3, 'first_3D_23.ply')
    err23,_ = projection.compute_error(norm_points23_3, hom_points_3d_23, P3)
    hom_points_3d_23 = hom_points_3d_23[err23 < err_threshold]

    ind23 = np.stack([np.arange(hom_points_3d_23.shape[0]), np.arange(hom_points_3d_23.shape[0])], axis=1)
    describe.draw_matching(data_filepath + '/0001.png', data_filepath + '/0002.png', \
                           np.fliplr(matched_points23_2[err23 < err_threshold]), np.fliplr(matched_points23_3[err23 < err_threshold]), ind23, \
                           matching_filepath + '/final-0001-0002.png', 'PNG')

    np.savetxt('fountain_dense/matching/matching_indices-0001-0002.txt', matches23[ratio23 < threshold][err23 < err_threshold])

    points_3D = np.concatenate([hom_points_3d, hom_points_3d_23], axis=0)
    points_3D = points_3D[:,:-1] / points_3D[:,-1, np.newaxis]
    points = np.array(list(zip(points_3D[:,0].ravel(), points_3D[:,1].ravel(), points_3D[:,2].ravel())),dtype=[('x','f4'), ('y','f4'),('z', 'f4')])
    el = reconstruct.PlyElement.describe(points, 'vertex')
    reconstruct.PlyData([el]).write('3D-1-2-3.ply')

    def add_image_to_reconstruction(img_name, prev_img_name, \
                                    desc_prev_with_known_3D, desc_prev_all, \
                                    keypoints_prev_with_known_3D, keypoints_prev_all, \
                                    prev_3D_points, P_prev, all_prev_3D_points):
                                    # keypoints_prev_with_known_3D must be in homogenous coordinates
        image = np.array(Image.open(data_filepath + img_name + '.png', 'r').convert('L')).astype(np.float).T
        keypoints = detect.harris_laplace(image, 7., 2e1, initial_sigma, steps_num, sigma_step)
        detect.draw_points_with_scale_markers(data_filepath + img_name + '.png', list(map(np.fliplr, keypoints)), (255,0,0), \
                                            keypoints_filepath + img_name + 'keypoints.png', 1.5, 1.2, \
                                            file_extension='PNG')
        sigmas = np.concatenate([np.ones(keypoints[i].shape[0]) * sigma_step ** i for i in range(len(keypoints))])
        keypoints = np.concatenate(keypoints, axis=0)
        desc = describe.compute_descriptors_for_all_points_with_scales(keypoints, image, sigmas)

        # desc23_2 = desc2[matches[ratio < threshold, 1], :][dists < outliers_threshold]

        matches, matches_dists, ratio = describe.match_keypoints(desc_prev_with_known_3D, desc)
        describe.remove_duplicates(matches, ratio)
        threshold = 0.8
        describe.draw_matching(data_filepath + prev_img_name + '.png', data_filepath + img_name + '.png', \
                            np.fliplr(keypoints_prev_with_known_3D[:, :-1]), np.fliplr(keypoints), matches[ratio < threshold], \
                            matching_filepath + prev_img_name + '-' + img_name[1:] + '-known3D.png', 'PNG')

        matched_points1 = keypoints_prev_with_known_3D[matches[ratio < threshold, 0], :]
        matched_points2 = keypoints[matches[ratio < threshold, 1], :]
        hom_points1 = matched_points1
        hom_points2 = to_homogenous(matched_points2)
        norm_points1 = np.linalg.inv(K).dot(hom_points1.T).T
        norm_points2 = np.linalg.inv(K).dot(hom_points2.T).T

        max_inliers = 0
        best_sample = np.empty(6, np.int)
        outliers_threshold = 1e-3

        iters = 10000
        for _ in range(iters):
            sample = np.random.choice(norm_points1.shape[0], 6, replace=False)

            P = projection.projection_matrix(norm_points2[sample], prev_3D_points[matches[ratio < threshold, 0], :][sample])
            err, err_sum = projection.compute_error(norm_points2, prev_3D_points[matches[ratio < threshold, 0], :], P)
            # print(err)
            if np.count_nonzero(err < outliers_threshold) > max_inliers:
                best_sample = sample
                max_inliers = np.count_nonzero(err < outliers_threshold)

        P = projection.projection_matrix(norm_points2[best_sample], prev_3D_points[matches[ratio < threshold, 0], :][best_sample])

        matches, matches_dists, ratio = describe.match_keypoints(desc_prev_all, desc)
        describe.remove_duplicates(matches, ratio)
        threshold = 0.8
        describe.draw_matching(data_filepath + prev_img_name + '.png', data_filepath + img_name + '.png', \
                            np.fliplr(keypoints_prev_all), np.fliplr(keypoints), matches[ratio < threshold], \
                            matching_filepath + prev_img_name + '-' + img_name[1:] + '-all.png', 'PNG')

        matched_points1 = keypoints_prev_all[matches[ratio < threshold, 0], :]
        matched_points2 = keypoints[matches[ratio < threshold, 1], :]
        hom_points1 = to_homogenous(matched_points1)
        hom_points2 = to_homogenous(matched_points2)
        norm_points1 = np.linalg.inv(K).dot(hom_points1.T).T
        norm_points2 = np.linalg.inv(K).dot(hom_points2.T).T

        err_threshold = 1e-5
        hom_points_3d = reconstruct.triangulate_points(P_prev, P, norm_points1, norm_points2, 'first_3D' + prev_img_name[1:] + '-' + img_name[1:] + '.ply')
        err,_ = projection.compute_error(norm_points2, hom_points_3d, P)
        hom_points_3d = hom_points_3d[err < err_threshold]

        ind = np.stack([np.arange(hom_points_3d.shape[0]), np.arange(hom_points_3d.shape[0])], axis=1)
        describe.draw_matching(data_filepath + prev_img_name + '.png', data_filepath + img_name + '.png', \
                            np.fliplr(matched_points1[err < err_threshold]), np.fliplr(matched_points2[err < err_threshold]), ind, \
                            matching_filepath + prev_img_name + '-' + img_name[1:] + '-final.png', 'PNG')
        np.savetxt('fountain_dense/matching/matching_indices' + prev_img_name[1:] + '-' + img_name[1:] + '.txt', matches[ratio < threshold][err < err_threshold])

        points_3d = hom_points_3d[:,:-1] / hom_points_3d[:,-1, np.newaxis]
        points_3D = np.concatenate([all_prev_3D_points, points_3d], axis=0)
        # points_3D = points_3D[:,:-1] / points_3D[:,-1, np.newaxis]
        points = np.array(list(zip(points_3D[:,0].ravel(), points_3D[:,1].ravel(), points_3D[:,2].ravel())),dtype=[('x','f4'), ('y','f4'),('z', 'f4')])
        el = reconstruct.PlyElement.describe(points, 'vertex')
        reconstruct.PlyData([el]).write('3D-all-to-' + img_name[1:] + '.ply')

        return desc[matches[ratio < threshold, 1], :][err < err_threshold], desc, \
               hom_points2[err < err_threshold], keypoints, \
               hom_points_3d, P, points_3D

    i = 3
    name = '/000' + str(i)
    prev_name = '/000' + str(i-1)
    desc_known_3D_temp, desc_temp, hom_points2_temp, keypoints_temp, hom_points_3d_temp, P_temp, points_3D_temp = \
        add_image_to_reconstruction(name, prev_name, desc3[matches23[ratio23 < threshold, 1], :][err23 < err_threshold], desc3, hom_points23_3[err23 < err_threshold], keypoints3, hom_points_3d_23, P3, points_3D)
    np.savetxt('fountain_dense/P_%d.txt' % i, P_temp)
    np.savetxt('fountain_dense/3D_to_%d.txt' % i, points_3D_temp)
    np.savetxt('fountain_dense/keypoints/descriptors_%d.txt' % i, desc_temp)
    np.savetxt('fountain_dense/keypoints/keypoints_%d.txt' % i, keypoints_temp)

    for i in range(4,11):
        print('Processing image nuber: {}.'.format(i))
        if i < 10:
            name = '/000' + str(i)
        else:
            name = '/00' + str(i)
        prev_name = '/000' + str(i-1)
        desc_known_3D_temp, desc_temp, hom_points2_temp, keypoints_temp, hom_points_3d_temp, P_temp, points_3D_temp = \
            add_image_to_reconstruction(name, prev_name, desc_known_3D_temp, desc_temp, hom_points2_temp, keypoints_temp, hom_points_3d_temp, P_temp, points_3D_temp)
        np.savetxt('fountain_dense/P_%d.txt' % i, P_temp)
        np.savetxt('fountain_dense/3D_to_%d.txt' % i, points_3D_temp)
        np.savetxt('fountain_dense/keypoints/descriptors_%d.txt' % i, desc_temp)
        np.savetxt('fountain_dense/keypoints/keypoints_%d.txt' % i, keypoints_temp)



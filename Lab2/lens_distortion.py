import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndim
from scipy.misc import imsave

def load_calib_data(filename):
    with open(filename, 'r') as f:
        [f1, s, f2, u, v] = f.readline().split()
        K = np.array( [[np.float(f1), np.float(s) , np.float(u)],
                       [0.0         , np.float(f2), np.float(v)],
                       [0.0         , 0.0         , 1.0        ]])
        f.readline() # read empty line
        distortion = np.array([np.float(x) for  x in f.readline().split()])
        f.readline() # read empty line

        extrinsic_matrices = []
        for _ in range(5):
            l = []
            for _ in range(3):
                l.append(f.readline().split())
            R = np.array(l).astype(np.float)
            t = np.array(f.readline().split()).astype(np.float)
            extrinsic_matrices.append(np.concatenate((R, t[:, np.newaxis]), axis=1))
            f.readline()
    
    return K, distortion, extrinsic_matrices


def load_corners_points(filename):
    with open(filename, 'r') as f:
        points = []
        for line in f:
            points.append(np.array(line.split()).astype(np.float))
    points = np.array(points).reshape((-1,2))
    return np.concatenate((points, np.zeros((points.shape[0],1))), axis=1)


def to_homogenous(data):
    return np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

def from_homogenous(data):
    return (data / data[:,-1, np.newaxis])[:,:-1]

def draw_image_with_points(filename, points, new_filename):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    width = 3
    for point in points:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = 200)
        draw.point(point[:-1], fill = 256)
    im.save(new_filename, 'GIF')

def distort_points(points, distortion_params):
    r_square = np.sum(points **2, axis=1)[:,np.newaxis]
    distorted_points = points * (1 + distortion_params[0] * r_square + distortion_params[1] * r_square * r_square)
    return distorted_points


def undistort_image(filename, distortion, intrinsic_matrix, extrinsic_matrix, new_filename):
    image = ndim.imread(filename)

    image0_points, image1_points = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    sample = np.stack((image0_points, image1_points), axis=2).reshape((-1,2))

    hom_sample = to_homogenous(sample)
    inv_K = np.linalg.inv(intrinsic_matrix)
    norm_sample = inv_K.dot(hom_sample.T).T
    distorted_sample = distort_points(from_homogenous(norm_sample), distortion)

    coordinates = from_homogenous(intrinsic_matrix.dot(to_homogenous(distorted_sample).T).T)
    coordinates = np.fliplr(coordinates)

    mappingR = ndim.map_coordinates(image[:,:,0], coordinates.T, order=1).reshape(image.shape[0:-1])
    mappingG = ndim.map_coordinates(image[:,:,1], coordinates.T, order=1).reshape(image.shape[0:-1])
    mappingB = ndim.map_coordinates(image[:,:,2], coordinates.T, order=1).reshape(image.shape[0:-1])

    new_image = np.stack((mappingR, mappingG, mappingB), axis=2)

    imsave(new_filename, new_image)

if __name__ == '__main__':
    K, distrt, extrinsic_matrices = load_calib_data('data/Calib.txt')
    points = load_corners_points('data/Model.txt')

    # task 1
    P1 = K.dot(extrinsic_matrices[0])
    P2 = K.dot(extrinsic_matrices[1])

    hom_points = to_homogenous(points)
    points2d = from_homogenous(P1.dot(hom_points.T).T)
    draw_image_with_points('data/UndistortIm1.gif', points2d, 'data/UndIm1withPoints.gif')
    draw_image_with_points('data/CalibIm1.gif', points2d, 'data/CalibIm1withPoints.gif')

    dist_points2d = distort_points(from_homogenous(extrinsic_matrices[0].dot(hom_points.T).T), distrt)
    p = to_homogenous(dist_points2d)
    p = K.dot(p.T)
    dist_points2d = from_homogenous(p.T)
    draw_image_with_points('data/CalibIm1.gif', dist_points2d, 'data/DistIm1withPoints.gif')



    points2d = from_homogenous(P2.dot(hom_points.T).T)
    draw_image_with_points('data/UndistortIm2.gif', points2d, 'data/UndIm2withPoints.gif')
    draw_image_with_points('data/CalibIm2.gif', points2d, 'data/CalibIm2withPoints.gif')

    dist_points2d = distort_points(from_homogenous(extrinsic_matrices[1].dot(hom_points.T).T), distrt)
    p = to_homogenous(dist_points2d)
    p = K.dot(p.T)
    dist_points2d = from_homogenous(p.T)
    draw_image_with_points('data/CalibIm2.gif', dist_points2d, 'data/DistIm2withPoints.gif')

    #task 2


    for i in range(5):
        undistort_image('data/CalibIm' + str(i+1) + '.gif', distrt, K, extrinsic_matrices[0], 'data/myUndistortedIm' + str(i+1) + '.gif')
        P = K.dot(extrinsic_matrices[i])
        hom_points = to_homogenous(points)
        points2d = from_homogenous(P.dot(hom_points.T).T)
        draw_image_with_points('data/myUndistortedIm' + str(i+1) + '.gif', points2d, 'data/myUndistortedIm' + str(i+1) + 'withPoints.gif')

    # task 3
    l = []
    lb = []
    for i in range(5):
        observed_points = load_corners_points('data/data' + str(i+1) + '.txt')
        P = K.dot(extrinsic_matrices[i])
        hom_points = to_homogenous(points)
        points2d = from_homogenous(P.dot(hom_points.T).T)
        center = K[0:2, 2]
        norm_points = from_homogenous(np.linalg.inv(K).dot(to_homogenous(points2d).T).T)
        x_y_square = np.array([np.sum(norm_points**2, axis=1), np.sum(norm_points**2, axis=1)**2]).T

        M = np.zeros((2*observed_points.shape[0],2))
        M[::2,:] = (points2d[:,0] - center[0])[:,np.newaxis] * x_y_square
        M[1::2,:] = (points2d[:,1] - center[1])[:,np.newaxis] * x_y_square

        b = np.zeros(2*observed_points.shape[0])
        b[::2] = observed_points[:,0] - points2d[:,0]
        b[1::2] = observed_points[:,1] - points2d[:,1]
        l.append(M)
        lb.append(b)

    A = np.concatenate(l, axis=0)
    b = np.concatenate(lb, axis=0)

    result,_,_,_ = np.linalg.lstsq(A, b)

from scipy import io
import numpy as np
from PIL import Image, ImageDraw
from Lab3.plyfile import PlyData, PlyElement


def get_linear_solution(points_image_1, points_image_2):
    points_number = points_image_1.shape[0]
    A = np.zeros((points_number, 9))
    A[:,:3] = points_image_1[:,0, np.newaxis] * points_image_2
    A[:,3:6] = points_image_1[:,1, np.newaxis] * points_image_2
    A[:,6:] = points_image_1[:,2, np.newaxis] * points_image_2

    _, _, V = np.linalg.svd(A, full_matrices=True)

    M = V[-1,:].reshape((3,3))

    return M

def fundamental_matrix(points_image_1, points_image_2):
    '''Function returns fundamental matrix 
    for x from image_2 and x' from image_1 '''
    F = get_linear_solution(points_image_1, points_image_2)
    u,s,v = np.linalg.svd(F)
    s[-1] = 0
    F = u.dot(np.diag(s).dot(v))

    return F

def essential_matrix(points_image_1, points_image_2, K):
    '''Function returns essential matrix 
    for x from image_2 and x' from image_1 '''
    inv_K = np.linalg.inv(K)
    norm_points1 = inv_K.dot(points_image_1.T).T
    norm_points2 = inv_K.dot(points_image_2.T).T
    E = get_linear_solution(norm_points1, norm_points2)
    u,s,v = np.linalg.svd(E)
    s = np.diag([(s[0] + s[1]) / 2., (s[0] + s[1]) / 2., 0.])
    E = u.dot(s.dot(v))
    return E

def get_dists_from_epipolar_lines(points_image_1, points_image_2, F):
    '''Get sum of distances from points of one image to theirs epipolar lines.
    Points should be in homogenous coordinates.'''
    epipolar_lines = F.dot(points_image_2.T).T
    dists = np.abs(np.sum(points_image_1 * epipolar_lines, axis=1)) / np.sqrt(epipolar_lines[:,0] ** 2 + epipolar_lines[:,1] ** 2)
    return dists


def draw_epipolar_line(point, F, draw, width):
    l = F.dot(point)
    l = l / (-l[1])
    draw.line((0, l[2],width, width * l[0] + l[2]), fill=128, width=2)
    return draw

def draw_epipolar_line_yx(point, F, draw, width):
    l = F.dot(point)
    l = l / (-l[0])
    draw.line((0, l[2],width, width * l[1] + l[2]), fill=128, width=2)
    return draw

def get_normalization_matrix(points):
    meanX = np.mean(points[:,0])
    meanY = np.mean(points[:,1])
    stdX = np.std(points[:,0])
    stdY = np.std(points[:,1])
    sqrt2 = np.sqrt(2)
    scale_matrix = np.array([
                [sqrt2 / stdX, 0,            0],
                [0,            sqrt2 / stdY, 0],
                [0,            0,            1]])
    translation_matrix = np.array([
                        [1, 0, -meanX],
                        [0, 1, -meanY],
                        [0, 0,  1    ]])
    T = scale_matrix.dot(translation_matrix)
    return T

def compute_3d_point(P1, P2, norm_point_1, norm_point_2):
    A = np.array([
        norm_point_1[0] * P1[2,:] - P1[0,:],
        norm_point_1[1] * P1[2,:] - P1[1,:],
        norm_point_2[0] * P2[2,:] - P2[0,:],
        norm_point_2[1] * P2[2,:] - P2[1,:]])

    _,_,v = np.linalg.svd(A)
    return v[-1,:]

def number_of_good_points(P, points):
    camera_Z_direction = P.dot(np.array([0., 0., 100000., 1.]))[2]
    number_of_good_points = np.count_nonzero(P.dot(points.T).T[:,2] * camera_Z_direction > 0)
    return number_of_good_points

def compare_matrix(P1, P2, points_image_1, points_image_2):
    hom_points_3d = np.array([compute_3d_point(P1, P2, point1, point2) for point1, point2 in zip(points_image_1, points_image_2)])
    points_3d = hom_points_3d / hom_points_3d[:,-1,np.newaxis]
    return number_of_good_points(P1, points_3d) + number_of_good_points(P2, points_3d)


def get_projection_matrices_from_essential(E, points_image_1, points_image_2):
    ''' Points sholud be normalized (multiplied by inverse of K) '''

    P1 = np.eye(3, M=4)
    U,S,Vt = np.linalg.svd(E)
    U *= S[0]
    S /= S[0]
    W = np.array([[0., -1., 0.],
                  [1.,  0., 0.],
                  [0.,  0., 1.]])
    u3 = U[:,2]
    P2s = [np.hstack((U.dot(W.dot(Vt)), u3[:,np.newaxis])),
           np.hstack((U.dot(W.dot(Vt)), -u3[:,np.newaxis])),
           np.hstack((U.dot(W.T.dot(Vt)), u3[:,np.newaxis])),
           np.hstack((U.dot(W.T.dot(Vt)), -u3[:,np.newaxis]))]
    P2 = P2s[np.argmax([compare_matrix(P1, P, points_image_1, points_image_2) for P in P2s])]

    return P1, P2

def triangulate_points(P1, P2, points_image_1, points_image_2, filename):
    ''' Points sholud be normalized (multiplied by inverse of K) '''
    hom_points_3d = np.array([compute_3d_point(P1, P2, point1, point2) for point1, point2 in zip(points_image_1, points_image_2)])
    points_3d = hom_points_3d / hom_points_3d[:,-1,np.newaxis]

    points = np.array(list(zip(points_3d[:,0].ravel(), points_3d[:,1].ravel(), points_3d[:,2].ravel())),dtype=[('x','f4'), ('y','f4'),('z', 'f4')])
    el = PlyElement.describe(points, 'vertex')
    PlyData([el]).write(filename)

if __name__ == '__main__':

    x = io.loadmat('data/compEx1data.mat')['x']

    points_image_1 = x[0,0].T
    points_image_2 = x[1,0].T
        
    #task 1

    F = fundamental_matrix(points_image_1, points_image_2)

    #task 2

    im = Image.open('data/kronan1.JPG')
    draw = ImageDraw.Draw(im)


    for i in range(20):
        draw = draw_epipolar_line(points_image_2[i,:], F, draw, im.size[0])
    width = 3
    fill = (0,255,0)
    for point in points_image_1[:20,:]:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save('data/kronan1_epipoles.JPG', 'JPEG')

    im = Image.open('data/kronan2.JPG')
    draw = ImageDraw.Draw(im)

    for i in range(20):
        draw = draw_epipolar_line(points_image_1[i,:], F.T, draw, im.size[0])
    for point in points_image_2[:20,:]:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save('data/kronan2_epipoles.JPG', 'JPEG')

    #task 3

    T1 = get_normalization_matrix(points_image_1)
    T2 = get_normalization_matrix(points_image_2)

    norm_points1 = T1.dot(points_image_1.T).T
    norm_points2 = T2.dot(points_image_2.T).T

    F = fundamental_matrix(norm_points1, norm_points2)

    F = T1.T.dot(F.dot(T2))

    im = Image.open('data/kronan1.JPG')
    draw = ImageDraw.Draw(im)


    for i in range(20):
        draw = draw_epipolar_line(points_image_2[i,:], F, draw, im.size[0])
    for point in points_image_1[:20,:]:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save('data/kronan1_norm_epipoles.JPG', 'JPEG')

    im = Image.open('data/kronan2.JPG')
    draw = ImageDraw.Draw(im)


    for i in range(20):
        draw = draw_epipolar_line(points_image_1[i,:], F.T, draw, im.size[0])
    for point in points_image_2[:20,:]:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save('data/kronan2_norm_epipoles.JPG', 'JPEG')

    # task 4

    K = io.loadmat('data/compEx3data.mat')['K']
    E = essential_matrix(points_image_1, points_image_2, K)
    new_F = np.linalg.inv(K.T).dot(E.dot(np.linalg.inv(K)))

    im = Image.open('data/kronan1.JPG')
    draw = ImageDraw.Draw(im)

    for i in range(20):
        draw = draw_epipolar_line(points_image_2[i,:], new_F, draw, im.size[0])
    for point in points_image_1[:20,:]:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save('data/kronan1_norm_ess_epipoles.JPG', 'JPEG')

    im = Image.open('data/kronan2.JPG')
    draw = ImageDraw.Draw(im)

    for i in range(20):
        draw = draw_epipolar_line(points_image_1[i,:], new_F.T, draw, im.size[0])
    for point in points_image_2[:20,:]:
        x = point[0]
        y = point[1]
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fill)
        draw.point(point[:-1], fill = fill)
    im.save('data/kronan2_norm_ess_epipoles.JPG', 'JPEG')


    #task 5

    norm_points1 = np.linalg.inv(K).dot(points_image_1.T).T
    norm_points2 = np.linalg.inv(K).dot(points_image_2.T).T
    P2, P1 = get_projection_matrices_from_essential(E, points_image_2, points_image_1)
    triangulate_points(P2, P1, norm_points2, norm_points1, 'pc.ply')
import numpy as np
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def projection_matrix(data2d, data3d):
    A = np.zeros((2*data2d.shape[0],12))
    A[::2,:4] = data3d
    A[1::2,4:8] = data3d
    A[::2,8:] = -data2d[:,0,np.newaxis] * data3d
    A[1::2,8:] = -data2d[:,1, np.newaxis] * data3d

    _, _, V = np.linalg.svd(A, full_matrices=True)
    
    P = V[-1,:].T.reshape((3,4))
    # print(P)
    return P

def compute_error(data2d, data3d, P):
    proj_data2d = np.dot(P, data3d.T).T
    proj_data2d = proj_data2d / proj_data2d[:,-1, np.newaxis]
    err = np.sum((proj_data2d[:,:-1] - data2d[:,:-1])**2,axis=1)
    err_sum = np.sum(err)
    return err, err_sum

def decomp_projection_matrix(P):
    K, R = linalg.rq(P[:,:-1])
    s = np.diag(np.sign(np.diag(K)))
    K = K.dot(s)
    R = R.dot(s)
    # print(s)
    T = np.linalg.inv(K).dot(P[:,-1])
    C = R.T.dot(-T)
    return K, R, T, C

if __name__ == '__main__':

    data2d = pd.read_csv('data/task12/pts2d-norm-pic_a.txt', header=None).values.astype(np.float)
    # print(data2d)

    data3d = pd.read_csv('data/task12/pts3d-norm.txt', header=None, sep='   ', engine='python').values.astype(np.float)
    # print(data3d)
    hom_data2d = np.concatenate((data2d, np.ones(data2d.shape[0])[:,np.newaxis]), axis=1)
    hom_data3d = np.concatenate((data3d, np.ones(data2d.shape[0])[:,np.newaxis]), axis=1)

    M = projection_matrix(hom_data2d, hom_data3d)
    _, err = compute_error(hom_data2d, hom_data3d, M)
    print('Error from normalized data: ', err)

    data2d = pd.read_csv('data/task12/pts2d-pic_a.txt', header=None, sep='  ', engine='python').values.astype(np.float)
    # print(data2d)

    data3d = pd.read_csv('data/task12/pts3d.txt', header=None, sep=' ').values.astype(np.float)
    # print(data3d)

    hom_data2d = np.concatenate((data2d, np.ones(data2d.shape[0])[:,np.newaxis]), axis=1)
    hom_data3d = np.concatenate((data3d, np.ones(data2d.shape[0])[:,np.newaxis]), axis=1)

    M = projection_matrix(hom_data2d, hom_data3d)
    _, err = compute_error(hom_data2d, hom_data3d, M)
    print('Error from not-normalized data: ', err)




# Put your imports here
import numpy as np
import math


def hartley_normalize(points):
    """
    Perform Hartley normalization on the given points.

    Args:
        points: (n x 3) numpy array, where n is the number of points. The points
        are in homogeneous coordinates.

    Returns:
        A tuple (normalized_points, T) where normalized_points is a (n x 2)
        numpy array which contains the 2D point coordinates after the
        normalization, and T is a (3 x 3) numpy array which performs the
        Hartley normalization on 2D points with homogeneous coordinates.
    """

    '''divide by w'''
    for point in range(points.shape[0]):
        points[point] /= points[point,2]

    points = points[:,:2]
    
    '''Part a, translate points so centroid is at origin'''
    #Calculate centroid
    xCent = np.mean(points[:,0])
    yCent = np.mean(points[:,1])
    posCent = np.array([xCent, yCent])

    #shift points to centroid
    points = points - posCent

    '''Part b, scale points so average distance from origin is sqrt(2)'''
    summation = 0
    for point in range(points.shape[0]):
        norm = math.sqrt(points[point,0]**2+points[point,1]**2)
        summation += norm
    avgDist = summation/points.shape[0]
    points[:,0] = points[:,0]/avgDist*math.sqrt(2)
    points[:,1] = points[:,1]/avgDist*math.sqrt(2)

    '''Part c, find T'''
    k = math.sqrt(2)/avgDist
    T = np. array([[1, 0, -xCent],
                   [0, 1, -yCent],
                   [0, 0,      1]])
    T2 = np. array([[k,0, 0],
                   [0, k, 0],
                   [0, 0, 1]])
    T = np.dot(T2,T)
    
    return points, T


def compute_fundamental_matrix(points1, points2):
    """
    Perform Hartley normalization on the given points.

    Args:
        points1: (n x 3) numpy array, where n is the number of points. The points
        are in homogeneous coordinates.
        points2: (n x 3) numpy array, where n is the number of points. The
        points are in homogeneous coordinates. Points in points1 and points2
        with the same index are in correspondence.

    Returns:
        A tuple (F, F_full) where F is the fundamental matrix for the original
        points (a (3 x 3) numpy array) and F_full is the fundamental matrix for
        the original points before rank adjustment, i.e. p2^T F_full p1 = 0
        should be guaranteed.
    """

    '''Part a, obtain T and T', normalize the input points'''
    p1,T1 = hartley_normalize(points1)
    p2,T2 = hartley_normalize(points2)

    '''Part b, build a linear equation system to solve for F_hat, each equation is p'T*F*p = 0'''
    A = np.empty((points1.shape[0],9))
    for n in range(points1.shape[0]):
        A[n,0] = p1[n,0]*p2[n,0]
        A[n,1] = p1[n,1]*p2[n,0]
        A[n,2] = p2[n,0]
        A[n,3] = p1[n,0]*p2[n,1]
        A[n,4] = p1[n,1]*p2[n,1]
        A[n,5] = p2[n,1]
        A[n,6] = p1[n,0]
        A[n,7] = p1[n,1]
        A[n,8] = 1

    '''Part c, To solve Af=0 you should take the right singular'''
    U,S,V = np.linalg.svd(A)
    f = V[-1]
    F_hat = np.reshape(f, (3,3))
    F_hat = np.dot(np.transpose(T2), np.dot(F_hat, T1))

    '''Part d, rank2'''
    U,S,V = np.linalg.svd(F_hat)
    S[2] = 0
    S_prime = np.diag(S)
    F_hat_prime = np.dot(U, np.dot(S_prime,V))
    
    '''Part e, reverse normalize'''
    F = np.dot(np.transpose(T2), np.dot(F_hat_prime, T1))

    return F, F_hat



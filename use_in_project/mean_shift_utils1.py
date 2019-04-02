from math import *
import numpy as np


def lon_lat_distance(pointA,pointB):
    if (len(pointA) != len(pointB)):
        print('Dimensions of A and B should be the same!')
    radius = 6378.137
    a = pointA[0] * acos(-1) / 180 - pointB[0] * acos(-1) / 180
    b = pointA[1] * acos(-1) / 180 - pointB[1] * acos(-1) / 180
    s = 2 * asin(sqrt((sin(a / 2) ** 2) + cos(pointA[1] * acos(-1) / 180) * cos(pointB[1] * acos(-1) / 180) * (sin(b / 2)** 2)))
    s = float(radius * s)

    return s

def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt((distance ** 2).sum(axis=1))
    val = (1/(bandwidth*sqrt(2*pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return val

def multivariate_gaussian_kernel(distances, bandwidths):

    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val


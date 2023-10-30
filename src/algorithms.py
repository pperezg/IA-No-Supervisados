'''
Paulina Pérez Garcés, 2023

algorithms.py contains the functions apply all clustering algorithms used for
the project.
'''

# Imports for this file
import numpy as np
import numpy.ma as ma
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from distances import *

'''
Function: pseudo_knn

Applies the k-nearest neighbors-like algorithm to the data. This is a
unsurpervised algorithm that groups the data in grouos of k points. It works
similarly to the KNN algorithm, but seeing as it is unsupervised, it does not
take into account the labels of the data points. This is a naïve clustering
algorithm created by the students of the AI course.

Parameters: norm (numpy array), k (int)
* norm: the norm to be applied to the data
* k: the number of neighbors to be considered

Returns: results (numpy array)
'''
def pseudo_knn(norm, k):

    #Decides how many neighborhoods there will be, according to k
    size = np.shape(norm)[0]
    if size%k == 0:
        neighborhoods = size//k
    else:
        neighborhoods = size//k + 1

    #Creates the results array
    results = np.zeros((size, neighborhoods))

    #Creates a mask that ignores all datapoints that have already been assigned
    masked_norm = np.ma.masked_equal(norm, 0)
    centerPoint = 0 #Sets the first reference poing
    k_loop = k #Sets the number of neighbors to be considered
    
    for i in range(neighborhoods):

        results[centerPoint, i] = 1 #Adds center point to neighborhood i
        if i==neighborhoods-1: #If it is the last neighborhood, sets k_loop
            k_loop = size%k

        for j in range(k_loop-1):
            minDist = np.argmin(masked_norm[centerPoint,:]) #Finds closest point
            results[minDist, i] = 1 #Adds closest point to neighborhood i
            masked_norm[:,minDist] = ma.masked #Masks closest point
            masked_norm[minDist,:] = ma.masked 

        nextCenterFound = False
        m = 0

        masked_norm[:,centerPoint] = ma.masked #Masks center point
        masked_norm[centerPoint,:] = ma.masked

        #Searches for a non-assigned point to be the next reference point
        while not nextCenterFound and m < size:
            if not masked_norm[m,:].mask.all():
                nextCenterFound = True
                centerPoint = m
            m+=1

    return results


'''
Function: boxes

Applies the boxes naïve algorithm to the data. This is a unsurpervised algorithm
that groups the data into spheres of a specified radius. This is a naïve
clustering algorithm created by the students of the AI course.

Parameters: norm (numpy array), distance (float)
* norm: the norm to be applied to the data
* distance: the radius of the spheres

Returns: results (numpy array)
'''
def boxes(norm, distance):

    size = np.shape(norm)[0] #Gets the size of the data
    # Creates a mask that ignores all datapoints that have already been assigned
    masked_norm = np.ma.masked_equal(norm, 0) 
    centerPoint = 0 #Sets the first reference point
    groups_bxs = [] #Creates a list to store the groups
    
    while not masked_norm.mask.all(): #While there are still unassigned points

        group_bxs = [] #Creates a list to store the current group
        group_bxs.append(centerPoint) #Adds the center point to the group
        for i in range(len(masked_norm[centerPoint,:])):
            #Appends all points within the distance to the group
            if masked_norm[centerPoint,i] <= distance:
                group_bxs.append(i)
                masked_norm[:,i] = ma.masked #Masks point
                masked_norm[i,:] = ma.masked
        groups_bxs.append(group_bxs) #Adds the group to the list of groups

        masked_norm[:,centerPoint] = ma.masked #Masks center point
        masked_norm[centerPoint,:] = ma.masked

        nextCenterFound = False

        m = 0
        #Searches for a non-assigned point to be the next reference point
        while not nextCenterFound and m < size:
            if not masked_norm[m,:].mask.all():
                nextCenterFound = True
                centerPoint = m
            m+=1

    num_groups = len(groups_bxs) #Gets the number of groups

    results = np.zeros((size, num_groups)) #Creates the results array

    #Adds the groups to the results array
    for i in range(len(groups_bxs)):
        for j in range(len(groups_bxs[i])):
            idx = groups_bxs[i][j]
            results[idx,i] = 1

    return results

def mountain(data, sigma, beta, K, norm, gridPoints=10):
    dims = data.shape[1]
    auxGrid = np.tile(np.linspace(0,1,gridPoints),dims).reshape(-1,gridPoints)
    grid = np.array(np.meshgrid(*auxGrid)).T.reshape(-1,dims)
    npoints = len(grid)

    #Initial center
    densities = np.zeros((npoints,1))
    for i in range(npoints):
        dist = applyNorm(norm, data, data2=grid[i,:])**2
        densities[i,0] = np.sum(np.exp(-dist/(2*sigma**2)))
    max_idx = np.argmax(densities)
    init_center = grid[max_idx,:]
    max_density = max(densities)[0]
    num_centers = 1

    #Find other centers
    centersArray = np.array([init_center])
    currentCenter = init_center

    while num_centers < K:
        aux_densities = np.zeros((npoints,1))
        dist2center = applyNorm(norm, grid, data2=centers[-1,:])**2
        aux_densities = densities[:,(num_centers-1)].reshape(-1,1) - max_density*np.exp(-dist2center/(2*beta**2)).reshape(-1,1)
        densities = np.append(densities,aux_densities,axis=1)
        max_density = max(densities[:,num_centers])
        max_idx = np.argmax(densities[:,num_centers])
        currentCenter = grid[max_idx,:]
        centersArray = np.append(centersArray, currentCenter.reshape(1,-1), axis = 0)
        num_centers += 1

    return centersArray

def subtractive(data, ra, factor_ra, K, norm):
    rb = ra*factor_ra
    dist = applyNorm(norm, data)**2
    n_data, dims = data.shape
    
    #Initial center
    densities = np.zeros((n_data,1))
    for i in range(n_data):
        densities[i,0] = np.sum(np.exp(-dist[i,:]/(ra**2)))
    max_idx = np.argmax(densities)
    init_center = data[max_idx,:]
    max_density = max(densities)[0]
    num_centers = 1

    #Find other centers
    centersArray = np.array([init_center])
    currentCenter = init_center
    repeats = 0
    while num_centers < K:
        aux_densities = densities[:,(num_centers-1)].reshape(-1,1) - max_density*np.exp(-dist[:,max_idx]/((0.5*rb)**2)).reshape(-1,1)
        densities = np.append(densities,aux_densities,axis=1)
        max_density = max(densities[:,num_centers])
        max_idx = np.argmax(densities[:,num_centers])
        currentCenter = data[max_idx,:]
        if currentCenter not in centersArray:
            centersArray = np.append(centersArray, currentCenter.reshape(1,-1), axis = 0)
        else:
            repeats += 1
            if repeats == 3:
                break
        num_centers += 1

    return centersArray






'''
KMEANS DE https://github.com/stuntgoat/kmeans/blob/master/kmeans.py
'''

from collections import defaultdict
from random import uniform
from math import sqrt

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in xrange(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.

    Compute the center for each of the assigned groups.

    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.itervalues():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = ()  # positive infinity
        shortest_index = 0
        for i in xrange(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    dimensions = len(a)
    
    _sum = 0
    for dimension in xrange(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in xrange(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in xrange(k):
        rand_point = []
        for i in xrange(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            
            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return zip(assignments, dataset)

'''
Fuzzy and Probabilistic C-Means de https://github.com/holtskinner/PossibilisticCMeans/blob/master/cmeans.py
'''

import numpy as np
from scipy.spatial.distance import cdist


def _eta(u, d, m):

    u = u ** m
    n = np.sum(u * d, axis=1) / np.sum(u, axis=1)

    return n


def _update_clusters(x, u, m):
    um = u ** m
    v = um.dot(x.T) / np.atleast_2d(um.sum(axis=1)).T
    return v


def _hcm_criterion(x, v, n, m, metric):

    d = cdist(x.T, v, metric=metric)

    y = np.argmin(d, axis=1)

    u = np.zeros((v.shape[0], x.shape[1]))

    for i in range(x.shape[1]):
        u[y[i]][i] = 1

    return u, d


def _fcm_criterion(x, v, n, m, metric):

    d = cdist(x.T, v, metric=metric).T

    # Sanitize Distances (Avoid Zeroes)
    d = np.fmax(d, np.finfo(x.dtype).eps)

    exp = -2. / (m - 1)
    d2 = d ** exp

    u = d2 / np.sum(d2, axis=0, keepdims=1)

    return u, d


def _pcm_criterion(x, v, n, m, metric):

    d = cdist(x.T, v, metric=metric)
    d = np.fmax(d, np.finfo(x.dtype).eps)

    d2 = (d ** 2) / n
    exp = 1. / (m - 1)
    d2 = d2.T ** exp
    u = 1. / (1. + d2)

    return u, d


def _cmeans(x, c, m, e, max_iterations, criterion_function, metric="euclidean", v0=None, n=None):

    if not x.any() or len(x) < 1 or len(x[0]) < 1:
        print("Error: Data is in incorrect format")
        return

    # Num Features, Datapoints
    S, N = x.shape

    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
    # If the user doesn't provide their own starting points,
    if v0 is None:
        # Pick random values from dataset
        xt = x.T
        v0 = xt[np.random.choice(xt.shape[0], c, replace=False), :]

    # List of all cluster centers (Bookkeeping)
    v = np.empty((max_iterations, c, S))
    v[0] = np.array(v0)

    # Membership Matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, c, N))

    # Number of Iterations
    t = 0

    while t < max_iterations - 1:

        u[t], d = criterion_function(x, v[t], n, m, metric)
        v[t + 1] = _update_clusters(x, u[t], m)

        # Stopping Criteria
        if np.linalg.norm(v[t + 1] - v[t]) < e:
            break

        t += 1

    return v[t], v[0], u[t - 1], u[0], d, t


# Public Facing Functions
def hcm(x, c, e, max_iterations, metric="euclidean", v0=None):
    return _cmeans(x, c, 1, e, max_iterations, _hcm_criterion, metric, v0=v0)


def fcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):

    return _cmeans(x, c, m, e, max_iterations, _fcm_criterion, metric, v0=v0)


def pcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):
    """

    Parameters
    ---

    `x` 2D array, size (S, N)
        Data to be clustered. N is the number of data sets;
        S is the number of features within each sample vector.

    `c` int
        Number of clusters

    `m` float, optional
        Fuzzifier

    `e` float, optional
        Convergence threshold

    `max_iterations` int, optional
        Maximum number of iterations

    `v0` array-like, optional
        Initial cluster centers

    Returns
    ---

    `v` 2D Array, size (S, c)
        Cluster centers

    `v0` 2D Array (S, c)
        Inital Cluster Centers

    `u` 2D Array (S, N)
        Final partitioned matrix

    `u0` 2D Array (S, N)
        Initial partition matrix

    `d` 2D Array (S, N)
        Distance Matrix

    `t` int
        Number of iterations run

    """

    v, v0, u, u0, d, t = fcm(x, c, m, e, max_iterations, metric=metric, v0=v0)
    n = _eta(u, d, m)
    return _cmeans(x, c, m, e, t, _pcm_criterion, metric, v0=v, n=n)


def rand(labels, cluster_results):
    return rand_score(labels, cluster_results)

def silhouette(unlabeled_data, cluster_results):
    return silhouette_score(unlabeled_data, cluster_results)
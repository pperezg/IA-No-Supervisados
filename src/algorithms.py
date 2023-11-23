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
from scipy.spatial.distance import cdist
from distances import *
from auxiliar import *

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
            if k_loop==0:
                k_loop=k

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
        dist2center = applyNorm(norm, grid, data2=centersArray[-1,:])**2
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


def kmeans(data, K, norm, max_iter=1000):
    centers = initialCenters(data, K)
    pastCenters = None
    iter = 1
    while iter < max_iter and np.not_equal(centers, pastCenters).any():
        distances = []
        for i in range(K):
            distances.append(applyNorm(norm, data, data2=centers[i,:]))
        labels = createClusters(data, centers, norm)
        pastCenters = centers
        centers = updateCenters(data, labels, K)
        iter += 1
    return labels
        

def initialCenters(data, K):
    n_data, dims = data.shape
    centers = np.zeros((K,dims))
    randomCenters = np.random.randint(0,n_data,K)
    for i in range(K):
        centers[i,:] = data[randomCenters[i],:]
    return centers

def updateCenters(data, labels, K):
    _, dims = data.shape
    centers = np.zeros((K,dims))
    for i in range(K):
        labelsMatch = np.equal(labels,i)
        dataInCenter = data[labelsMatch,:]
        centers[i,:] = np.mean(dataInCenter, axis=0)
    return centers








'''
Fuzzy and Probabilistic C-Means de https://github.com/holtskinner/PossibilisticCMeans/blob/master/cmeans.py
'''


def _eta(u, d, m):

    u = u ** m
    n = np.sum(u * d, axis=1) / np.sum(u, axis=1)

    return n


def _update_clusters(x, u, m):
    um = u ** m
    v = um.dot(x.T) / np.atleast_2d(um.sum(axis=1)).T
    return v

def _fcm_criterion(x, v, n, m, metric):

    d = applyNorm(metric, x.T, data2=v).squeeze().T
    d = d
    try:
        d = np.fmax(d, np.finfo(x.dtype).eps)
    except:
        pass

    exp = -2. / (m - 1)
    d2 = d ** exp

    u = d2 / np.sum(d2, axis=0, keepdims=1)

    return u, d


def _pcm_criterion(x, v, n, m, metric):

    d = applyNorm(metric, x.T, data2=v).squeeze()
    d = d
    try:
        d = np.fmax(d, np.finfo(x.dtype).eps)
    except:
        pass

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
    try:
        return rand_score(labels, cluster_results)
    except:
        return 0

def silhouette(unlabeled_data, cluster_results):
    try:
        return silhouette_score(unlabeled_data, cluster_results)
    except:
        return -1
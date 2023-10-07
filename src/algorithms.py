'''
Paulina Pérez Garcés, 2023

algorithms.py contains the functions apply all clustering algorithms used for
the project.
'''

# Imports for this file
import numpy as np
import numpy.ma as ma


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
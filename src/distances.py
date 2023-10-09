'''
Paulina Pérez Garcés, 2023

distances.py contains the functions to apply, calculate, and graph the different
norms used in the project.
'''

# Imports for this file
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


'''
Function: applyNorm

Calculates the distance matrixes between all data points in the data array
according to the specified norm.

Parameters: norm (string), data (numpy array), p (int), graph (boolean)
* norm: the norm to be applied to the data
* data: the data to which the norm will be applied
* p: the p value for the lp norm
* graph: whether or not to graph the result

Returns: result (numpy array)
'''
def applyNorm(norm, data, data2=None, p=6, graph=False):

    #Reshapes the data in order to use vectorize
    if data2==None:
        a = data[:,None,:]
        b = data[None,:,:]
    else:
        a = data[:,None,:]
        b = data2[None,:,:]

    #Vectorizes the norm function according to the parameter 'norm'
    if norm == 'euclid':
        fv = np.vectorize(euclideanNorm, signature='(d),(d)->()')
    elif norm == 'manhattan':
        fv = np.vectorize(manhattanNorm, signature='(d),(d)->()')
    elif norm == 'mahalanobis':
        fv = np.vectorize(mahalanobisNorm, signature='(d),(d)->()')
    elif norm == 'cosine':
        fv = np.vectorize(cosineNorm, signature='(d),(d)->()')
    elif norm == 'lp':
        fv = np.vectorize(lpNorm, signature='(d),(d)->()', excluded=['p'])

    #Applies the vectorized function to the data
    if norm=='lp':
        result = fv(a,b,p=p)
    else:
        result = fv(a,b)

    #If graph is True, graphs the result in a heatmap
    if graph:
        graphNorm(norm, result)

    return result


'''
Function: euclideanNorm

Calculates the euclidean distance between two data points.

Parameters: data1 (numpy array), data2 (numpy array)
* data1: the first data point
* data2: the second data point

Returns: the euclidean distance between data1 and data2
'''
def euclideanNorm(data1, data2):
    return np.sqrt(np.sum((data1 - data2)**2))


'''
Function: manhattanNorm

Calculates the manhattan distance between two data points.

Parameters: data1 (numpy array), data2 (numpy array)
* data1: the first data point
* data2: the second data point

Returns: the manhattan distance between data1 and data2
'''
def manhattanNorm(data1, data2):
    return np.sum(np.abs(data1 - data2))


'''
Function: mahalanobisNorm

Calculates the mahalanobis distance between two data points.

Parameters: data1 (numpy array), data2 (numpy array)
* data1: the first data point
* data2: the second data point

Returns: the mahalanobis distance between data1 and data2
'''
def mahalanobisNorm(data1, data2):
    return np.sqrt(np.sum((data1 - data2)**2))


'''
Function: cosineNorm

Calculates the cosine pseudonorm between two data points.

Parameters: data1 (numpy array), data2 (numpy array)
* data1: the first data point
* data2: the second data point

Returns: the cosine pseudonorm between data1 and data2
'''
def cosineNorm(data1, data2):
    return np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))


'''
Function: lpNorm

Calculates the distance between two data points according to the lp norm.

Parameters: data1 (numpy array), data2 (numpy array), p (int)
* data1: the first data point
* data2: the second data point
* p: the p value for the lp norm

Returns: the lp distance between data1 and data2
'''
def lpNorm(data1, data2, p):
    return np.sum(np.abs(data1 - data2)**p)**(1/p)


'''
Function: graphNorm

Creates a heatmap of the distances between all datapoints according to the
given norm. Displays the graph.

Parameters: norm (string), result (numpy array)
* norm: the norm used to calculate the result
* result: the result of the norm
'''
def graphNorm(norm, result):
    plt.imshow(result, cmap='autumn', interpolation='nearest')
 
    plt.title("Distances acording to "+norm+" norm.")
    plt.show()
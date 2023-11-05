'''
Paulina Pérez Garcés, 2023

data.py contains the functions to import, preprocess, characterize, and save
the data used for and produced by the project.
'''

#Imports for this file
from sklearn import datasets
from sklearn.preprocessing import normalize
import pandas as pd 
import numpy as np
from datetime import datetime


'''
Function: getData

Asks the user if they want to use the Iris dataset or their own dataset, and
returns the data and labels. In case of going for a personal dataset it deals
with all NaN values and one hot encodes the categorical variables. For both
datasets it normalizes the data.

Returns: features (numpy array), target (numpy array)
'''
def getData():

    #Ask used if they want to use the Iris dataset or their own
    print('Do you want to use "Iris" dataset? (y/n)')
    ans = input()

    #If the user enters something different than 'y' or 'n', asks again
    while ans != 'y' and ans != 'n':
        print('Please, enter "y" or "n"')
        ans = input()

    #If the user wants to use the Iris dataset, loads it
    if ans == 'y':
        iris_features = datasets.load_iris().data
        iris_target = datasets.load_iris().target
        #Normalizes the data
        iris_features = normalize(iris_features, norm='max', axis=0)
        
        return iris_features, iris_target

    #If the user wants to use their own dataset, asks for the path
    else:
        #Asks for the path
        print('Please, enter the path to your dataset (last column must contain the labels):')
        path = input()

        #If the path indicates dataset is in json format, reads it as json
        if path.endswith('.json'):
            data = pd.read_json(path)

        #If the path indicates dataset is in csv or txt format, reads it as csv
        else:
            data = pd.read_csv(path, header='infer')

        if data.isnull().values.any(): #Checks for NaN values
            print('Warning: There are NaN values in your dataset.')
            print('Do you want to remove columns (c) or rows (r) with NaN values? (c/r)')
            ans2 = input()

            while ans2 != 'c' and ans2 != 'r':
                print('Please, enter "c" or "r"')
                ans2 = input()

            if ans2 == 'c': #Removes columns with NaN values
                data = data.dropna(axis=1, how='all')
            else: #Removes rows with NaN values
                data = data.dropna(axis=0, how='all')

        #Divides the data into features and labels
        features = data.iloc[:,:-1]
        target = data.iloc[:,-1]

        # One hot encoding
        cat_cols = features.select_dtypes(include=['object']).columns.tolist()
        features = pd.get_dummies(features, columns = cat_cols)

        #Converts data to numpy arrays
        features = features.to_numpy()
        target = target.to_numpy()

        data = normalize(data, norm='max', axis=0) #Normalizes the data

        return features, target


'''
Function: dataCharacterization

Characterizes the data by returning the number of data points, the number of
characteristics, and the number of classes.

Parameters: features (numpy array), labels (numpy array)
* features: the data points
* labels: the labels of the data points

Returns: num_dataPoints (int), num_characteristics (int), num_classes (int)
'''
def dataCharacterization(features, labels):

    num_dataPoints, num_characteristics = np.shape(features)
    num_classes = len(np.unique(labels))

    return num_dataPoints, num_characteristics, num_classes


'''
Function: saveResultFile

Saves the results of the project in a txt file.

Parameters: cajas_df (list of numpy arrays), knn_df (list of numpy arrays),
norms (list of strings), N (int), n (int), m (int)
* cajas_df: results for the boxes naïve algorithm
* knn_df: results for the KNN-like naïve algorithm
* norms: list of the norms used
* N: number of data points
* n: number of characteristics
* m: number of classes
'''
def saveResultFile(cajas_df, knn_df, norms, N, n, m, name, silhouette, rand):
    
    #Gets the current date and time, sets results path accordingly
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    resultPath = 'src/results/results_' + name + '_'+ dt_string + '.txt'

    #Opens the file and writes the results
    with open(resultPath, 'w') as f:

        f.write('Number of data points: '+str(N)+'\n')
        f.write('Number of characteristics: '+str(n)+'\n')
        f.write('Number of classes: '+str(m)+'\n\n')

        for i in range(len(norms)):

            f.write('Norm: '+norms[i]+'\n\n')

            f.write('Boxes:\n')
            f.write('Silhouette score: '+str(silhouette[0][i])+'\n')
            f.write('Rand score: '+str(rand[0][i])+'\n')
            aux = pd.DataFrame(cajas_df[i])
            f.write(aux.to_string())
            f.write('\n')

            f.write('\nKNN:\n')
            f.write('Silhouette score: '+str(silhouette[1][i])+'\n')
            f.write('Rand score: '+str(rand[1][i])+'\n')
            aux = pd.DataFrame(knn_df[i])
            f.write(aux.to_string())
            f.write('\n\n\n\n\n')

        f.write('By: Paulina Pérez Garcés')

    print('Results saved in '+resultPath) #Prints the path to the results file


def saveResultFileMtn(combinatoria, name, silhouette, rand, best):
    
    #Gets the current date and time, sets results path accordingly
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    resultPath = 'src/results/results_Mtn_' + name + '_'+ dt_string + '.txt'

    #Opens the file and writes the results
    with open(resultPath, 'w') as f:

        f.write('Best Centers: '+str(best)+'\n\n')
        f.write('Sigma, Number of Clusters, Grid Points, Norm --> Silhouette, Rand' +'\n')
        for i in range(len(combinatoria)):
            f.write('Values: '+str(combinatoria[i])+' --> ')
            f.write('Silhouette: '+str(silhouette[i])+' ')
            f.write('Rand: '+str(rand[i])+'\n')

        f.write('\n By: Paulina Pérez Garcés')

    print('Results saved in '+resultPath) #Prints the path to the results file



def saveResultFileSub(combinatoria, name, silhouette, rand, best):
    
    #Gets the current date and time, sets results path accordingly
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    resultPath = 'src/results/results_Sub_' + name + '_'+ dt_string + '.txt'

    #Opens the file and writes the results
    with open(resultPath, 'w') as f:

        f.write('Best Centers: '+str(best)+'\n\n')
        f.write('Ra, Number of Clusters, Norm --> Silhouette, Rand' +'\n')
        for i in range(len(combinatoria)):
            f.write('Values: '+str(combinatoria[i])+' --> ')
            f.write('Silhouette: '+str(silhouette[i])+' ')
            f.write('Rand: '+str(rand[i])+'\n')

        f.write('\n By: Paulina Pérez Garcés')

    print('Results saved in '+resultPath) #Prints the path to the results file


def saveResultFileKmeans(combinatoria, name, silhouette, rand):
    #Gets the current date and time, sets results path accordingly
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    resultPath = 'src/results/results_KMeans_' + name + '_'+ dt_string + '.txt'

    bestRand = np.max(rand)
    bestSil = np.max(silhouette)

    #Opens the file and writes the results
    with open(resultPath, 'w') as f:

        f.write('Best Rand: '+str(bestRand)+'\n')
        f.write('Best Silhouette: '+str(bestSil)+'\n\n')
        f.write('K, Norm --> Silhouette, Rand' +'\n')
        for i in range(len(combinatoria)):
            f.write('Values: '+str(combinatoria[i])+' --> ')
            f.write('Silhouette: '+str(silhouette[i])+' ')
            f.write('Rand: '+str(rand[i])+'\n')

        f.write('\n By: Paulina Pérez Garcés')

    print('Results saved in '+resultPath) #Prints the path to the results file
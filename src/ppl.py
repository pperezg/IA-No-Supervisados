'''
Paulina Pérez Garcés

main.py is the main file of the project. It imports all the necessary modules 
and runs the algorithms. It also creates a virtual environment and installs all
the necessary packages.
'''

if __name__ == "__main__":

    #Import of the necessary modules from other files
    from algorithms import *
    from distances import *
    from data import *
    from autoencoder_UMAP import *
    from aux import *

    features, target = getData() #Get and organizes the data
    higher_dim = get_MLP_encoding(features) #Gets MLP encoding to higher dims
    umap2d = umap2d3d(features, 2) #Gets UMAP encoding to 2 dims
    umap3d = umap2d3d(features, 3) #Gets UMAP encoding to 3 dims
    N, n, m = dataCharacterization(features, target) #Gets data characteristics
    norms = ['euclid', 'manhattan', 'mahalanobis', 'cosine', 'lp'] #Norms used

    #Lists to store the results of the algorithms
    cajas_df = []
    knn_df = []
    silhouetteNaive = []
    randNaive = []

    for norma in norms:
        print('currently running norm: '+norma)
        norm = applyNorm(norma, features) #Applies the norm to the data
        groups_cajas = boxes(norm, 0.3) #Applies the boxes algorithm
        groups_knn = pseudo_knn(norm, 25) #Applies the knn-like algorithm
        cajas_df.append(groups_cajas) #Stores the results
        knn_df.append(groups_knn)
        labelsCajas = olga2Labels(groups_cajas)
        labelsKnn = olga2Labels(groups_knn)
    
    #Saves the results to .txt files
    saveResultFile(cajas_df, knn_df, norms, N, n, m, 'Naive', silhouetteNaive, randNaive)

    print('Done')
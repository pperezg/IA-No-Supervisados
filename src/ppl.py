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
    higher_dim = get_MLP_encoding(features).to_numpy() #Gets MLP encoding to higher dims
    umap2d = umap2d3d(features, 2) #Gets UMAP encoding to 2 dims
    N, n, m = dataCharacterization(features, target) #Gets data characteristics
    norms = ['euclid', 'manhattan', 'mahalanobis', 'lp'] #Norms used

    value_sets = [features, higher_dim, umap2d] #List of data sets

    #Lists to store the results of the algorithms
    cajas_df = []
    knn_df = []
    silhouetteNaive = [[],[]]
    randNaive = [[],[]]

    for data_set in value_sets:
        print('New data set')
        for norma in norms:
            print('currently running norm: '+norma)
            norm = applyNorm(norma, data_set) #Applies the norm to the data
            maxDistance = np.amax(norm) #Gets the max distance
            groups_cajas = boxes(norm, maxDistance*0.45) #Applies the boxes algorithm
            groups_knn = pseudo_knn(norm, 50) #Applies the knn-like algorithm
            cajas_df.append(groups_cajas) #Stores the results
            knn_df.append(groups_knn)
            labelsCajas = olga2Labels(groups_cajas)
            labelsKnn = olga2Labels(groups_knn)
            silhouetteNaive[0].append(silhouette_score(data_set, labelsCajas))
            silhouetteNaive[1].append(silhouette_score(data_set, labelsKnn))
            randNaive[0].append(rand_score(target, labelsCajas))
            randNaive[1].append(rand_score(target, labelsKnn))

            if data_set is umap2d:
                plot2dClusters(data_set, labelsCajas, 'Naive boxes', norma)
                plot2dClusters(data_set, labelsKnn, 'Pseudo KNN', norma)
        
        #Saves the results to .txt files
        if data_set is features:
            saveResultFile(cajas_df, knn_df, norms, N, n, m, 'naive_normalDims', silhouetteNaive, randNaive)
        elif data_set is higher_dim:
            saveResultFile(cajas_df, knn_df, norms, N, n, m, 'naive_higherDims', silhouetteNaive, randNaive)
        elif data_set is umap2d:
            saveResultFile(cajas_df, knn_df, norms, N, n, m, 'naive_umap2d', silhouetteNaive, randNaive)

    print('Done')
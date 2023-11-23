'''
Paulina Pérez Garcés

ppl.py is the file that ties all necessary procedures together. If no virtual
environment were needed, it would serve as the main file in the project.
'''

if __name__ == "__main__":

    #Import of the necessary modules from other files
    from algorithms import *
    from distances import *
    from data import *
    from autoencoder_UMAP import *
    from aux import *
    import itertools

    features, target = getData() #Get and organizes the data
    _, dataDims = features.shape #Gets the dimensions of the data
    higher_dim = get_MLP_encoding(features, dataDims).to_numpy() #Gets MLP encoding to higher dims
    umap2d = umap2d3d(features, 2) #Gets UMAP encoding to 2 dims
    N, n, m = dataCharacterization(features, target) #Gets data characteristics
    norms = ['euclid', 'manhattan', 'mahalanobis', 'lp'] #Norms used

    value_sets = [features, higher_dim, umap2d] #List of data sets
    ogClusters = len(set(target)) #Number of original clusters


    print('Evaluating Naive Models')

    #Lists to store the results of the algorithms
    cajas_df = []
    knn_df = []

    for data_set in value_sets:
        print('New data set')
        silhouetteNaive = [[],[]]
        randNaive = [[],[]]
        for norma in norms:
            print('currently running norm: '+norma)
            norm = applyNorm(norma, data_set) #Applies the norm to the data
            maxDistance = np.amax(norm) #Gets the max distance
            groups_cajas = boxes(norm, maxDistance*0.45) #Applies the boxes algorithm
            groups_knn = pseudo_knn(norm, 50) #Applies the knn-like algorithm
            #Stores the results
            cajas_df.append(groups_cajas) 
            knn_df.append(groups_knn)
            #Turns results into the desired output format
            labelsCajas = olga2Labels(groups_cajas)
            labelsKnn = olga2Labels(groups_knn)
            #Calculates the silhouette and rand scores
            silhouetteNaive[0].append(silhouette(data_set, labelsCajas))
            silhouetteNaive[1].append(silhouette(data_set, labelsKnn))
            randNaive[0].append(rand(target, labelsCajas))
            randNaive[1].append(rand(target, labelsKnn))

            if data_set is umap2d: #If dataset is 2d, graphs the results
                plot2dClusters(data_set, labelsCajas, 'Naive boxes', norma)
                plot2dClusters(data_set, labelsKnn, 'Pseudo KNN', norma)
        
        #Saves the results to .txt files
        if data_set is features:
            saveResultFile(cajas_df, knn_df, norms, N, n, m, 'naive_normalDims', silhouetteNaive, randNaive)
        elif data_set is higher_dim:
            saveResultFile(cajas_df, knn_df, norms, N, n, m, 'naive_higherDims', silhouetteNaive, randNaive)
        elif data_set is umap2d:
            saveResultFile(cajas_df, knn_df, norms, N, n, m, 'naive_umap2d', silhouetteNaive, randNaive)       

    

    print('Evaluating cluster centers: Mountain')

    #Parameters for the mountain algorithm
    sigma = [0.1,0.2]
    numClusters = [ogClusters-1,ogClusters,ogClusters+1]
    gridPoints = [2,3]
    silhouettes = []
    randScores = []
    centersArray = []
    combinationsMtn = list(itertools.product(sigma, numClusters, gridPoints, norms))

    print('Normal Dimensions')


    for i in range(len(combinationsMtn)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsMtn)))
        centers = mountain(features, combinationsMtn[i][0], combinationsMtn[i][0]*1.5, combinationsMtn[i][1], combinationsMtn[i][3], gridPoints=combinationsMtn[i][2])
        centersArray.append(centers) #Stores the centers
        labels = createClusters(features, centers, combinationsMtn[i][3]) #Creates the clusters
        #Calculates the silhouette and rand scores
        silhouetteMtn = silhouette(features, labels); silhouettes.append(silhouetteMtn) 
        randMtn = rand(target, labels); randScores.append(randMtn)

    #Finds iteration with the best max silhouette and max rand scores
    maxSilhouette = np.array(silhouettes).argmax()
    maxRand = np.array(randScores).argmax()
    if maxSilhouette==maxRand: #If both are the same, chooses that iteration
        bestNormalDims = centersArray[maxSilhouette]
    else: #If not, chooses the one with the best silhouette score
        auxSilhouette = silhouettes[maxSilhouette]
        auxRand = silhouettes[maxRand]
        auxMax = max(auxSilhouette, auxRand)
        if auxMax==auxSilhouette:
            bestNormalDims = centersArray[maxSilhouette]
        else:
            bestNormalDims = centersArray[maxRand]
    saveResultFileMtn(combinationsMtn, 'NormalDims', silhouettes, randScores, bestNormalDims) #Saves the results to .txt files
    silhouettes = []
    randScores = []
    centersArray = []

    print('UMAP') #Repetition of the same procedure for dataset in 2D

    for i in range(len(combinationsMtn)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsMtn)))
        centers = mountain(umap2d, combinationsMtn[i][0], combinationsMtn[i][0]*1.5, combinationsMtn[i][1], combinationsMtn[i][3], gridPoints=combinationsMtn[i][2])
        centersArray.append(centers)
        labels = createClusters(umap2d, centers, combinationsMtn[i][3])
        silhouetteMtn = silhouette(umap2d, labels); silhouettes.append(silhouetteMtn)
        randMtn = rand(target, labels); randScores.append(randMtn)
        imgName = 'Mountain'+str(combinationsMtn[i][:-1])
        plot2dClusters(umap2d, labels, imgName, combinationsMtn[i][3])

    maxSilhouette = np.array(silhouettes).argmax()
    maxRand = np.array(randScores).argmax()
    if maxSilhouette==maxRand:
        bestUMAP = centersArray[maxSilhouette]
    else:
        auxSilhouette = silhouettes[maxSilhouette]
        auxRand = silhouettes[maxRand]
        auxMax = max(auxSilhouette, auxRand)
        if auxMax==auxSilhouette:
            bestUMAP = centersArray[maxSilhouette]
        else:
            bestUMAP = centersArray[maxRand]
    saveResultFileMtn(combinationsMtn, 'UMAP', silhouettes, randScores, bestUMAP)    
    silhouettes = []
    randScores = []
    centersArray = []

    print('Higher Dimensions') #Repeticion of the same procedure for dataset in higher dimensions

    for i in range(len(combinationsMtn)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsMtn)))
        centers = mountain(higher_dim, combinationsMtn[i][0], combinationsMtn[i][0]*1.5, combinationsMtn[i][1], combinationsMtn[i][3], gridPoints=combinationsMtn[i][2])
        centersArray.append(centers)
        labels = createClusters(higher_dim, centers, combinationsMtn[i][3])
        silhouetteMtn = silhouette(higher_dim, labels); silhouettes.append(silhouetteMtn)
        randMtn = rand(target, labels); randScores.append(randMtn)

    maxSilhouette = np.array(silhouettes).argmax()
    maxRand = np.array(randScores).argmax()
    if maxSilhouette==maxRand:
        bestHigherDims = centersArray[maxSilhouette]
    else:
        auxSilhouette = silhouettes[maxSilhouette]
        auxRand = silhouettes[maxRand]
        auxMax = max(auxSilhouette, auxRand)
        if auxMax==auxSilhouette:
            bestHigherDims = centersArray[maxSilhouette]
        else:
            bestHigherDims = centersArray[maxRand]
    saveResultFileMtn(combinationsMtn, 'HigherDims', silhouettes, randScores, bestHigherDims)
    silhouettes = []
    randScores = []
    centersArray = []

    print('Evaluating cluster centers: Subtractive')

    #Parameters for the subtractive algorithm
    ra = [0.1,0.3,0.5,0.7,0.9]
    combinationsSub = list(itertools.product(ra, numClusters, norms))
    silhouettes = []
    randScores = []
    centersArray = []

    print('Normal Dimensions')

    for i in range(len(combinationsSub)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsSub)))
        centers = subtractive(features, combinationsSub[i][0], 1.5, combinationsSub[i][1], combinationsSub[i][2])
        centersArray.append(centers) #Stores the centers
        labels = createClusters(features, centers, combinationsSub[i][2]) #Creates the clusters
        #Calculates the silhouette and rand scores
        silhouetteMtn = silhouette(features, labels); silhouettes.append(silhouetteMtn)
        randMtn = rand(target, labels); randScores.append(randMtn)

    #Finds iteration with the best max silhouette and max rand scores
    maxSilhouette = np.array(silhouettes).argmax()
    maxRand = np.array(randScores).argmax()
    if maxSilhouette==maxRand: #If both are the same, chooses that iteration
        bestNormalDims = centersArray[maxSilhouette]
    else: #If not, chooses the one with the best silhouette score
        auxSilhouette = silhouettes[maxSilhouette]
        auxRand = silhouettes[maxRand]
        auxMax = max(auxSilhouette, auxRand)
        if auxMax==auxSilhouette:
            bestNormalDims = centersArray[maxSilhouette]
        else:
            bestNormalDims = centersArray[maxRand]
    saveResultFileSub(combinationsSub, 'NormalDims', silhouettes, randScores, bestNormalDims) #Saves the results to .txt files
    silhouettes = []
    randScores = []
    centersArray = []  

    print('UMAP') #Repetition of the same procedure for dataset in 2D

    for i in range(len(combinationsSub)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsSub)))
        centers = subtractive(umap2d, combinationsSub[i][0], 1.5, combinationsSub[i][1], combinationsSub[i][2])
        centersArray.append(centers)
        labels = createClusters(umap2d, centers, combinationsSub[i][2])
        silhouetteMtn = silhouette(umap2d, labels); silhouettes.append(silhouetteMtn)
        randMtn = rand(target, labels); randScores.append(randMtn)
        imgName = 'Subtractive'+str(combinationsSub[i][:-1])
        plot2dClusters(umap2d, labels, imgName, combinationsSub[i][2])

    maxSilhouette = np.array(silhouettes).argmax()
    maxRand = np.array(randScores).argmax()
    if maxSilhouette==maxRand:
        bestNormalDims = centersArray[maxSilhouette]
    else:
        auxSilhouette = silhouettes[maxSilhouette]
        auxRand = silhouettes[maxRand]
        auxMax = max(auxSilhouette, auxRand)
        if auxMax==auxSilhouette:
            bestNormalDims = centersArray[maxSilhouette]
        else:
            bestNormalDims = centersArray[maxRand]
    saveResultFileSub(combinationsSub, 'UMAP', silhouettes, randScores, bestNormalDims)
    silhouettes = []
    randScores = []
    centersArray = []   

    print('Higher Dimensions') #Repeticion of the same procedure for dataset in higher dimensions

    for i in range(len(combinationsSub)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsSub)))
        centers = subtractive(higher_dim, combinationsSub[i][0], 1.5, combinationsSub[i][1], combinationsSub[i][2])
        centersArray.append(centers)
        labels = createClusters(higher_dim, centers, combinationsSub[i][2])
        silhouetteMtn = silhouette(higher_dim, labels); silhouettes.append(silhouetteMtn)
        randMtn = rand(target, labels); randScores.append(randMtn)

    maxSilhouette = np.array(silhouettes).argmax()
    maxRand = np.array(randScores).argmax()
    if maxSilhouette==maxRand:
        bestNormalDims = centersArray[maxSilhouette]
    else:
        auxSilhouette = silhouettes[maxSilhouette]
        auxRand = silhouettes[maxRand]
        auxMax = max(auxSilhouette, auxRand)
        if auxMax==auxSilhouette:
            bestNormalDims = centersArray[maxSilhouette]
        else:
            bestNormalDims = centersArray[maxRand]
    saveResultFileSub(combinationsSub, 'HigherDims', silhouettes, randScores, bestNormalDims)
    silhouettes = []
    randScores = []
    centersArray = []  

    print('Evaluating clustering algorithms: KMeans') 
    # Parameters for the KMeans algorithm
    K = [ogClusters-1,ogClusters,ogClusters+1]
    combinationsKMeans = list(itertools.product(K, norms))
    silhouettes = []
    randScores = []
    
    print('Normal Dimensions')

    for i in range(len(combinationsKMeans)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsKMeans)))
        labels = kmeans(features, combinationsKMeans[i][0], combinationsKMeans[i][1]) #Applies the KMeans algorithm
        #Calculates the silhouette and rand scores
        silhouetteKNN = silhouette(features, labels); silhouettes.append(silhouetteKNN)
        randKNN = rand(target, labels); randScores.append(randKNN)
    saveResultFileKmeans(combinationsKMeans, 'NormalDims', silhouettes, randScores) #Saves the results to .txt files

    print('UMAP') #Repetition of the same procedure for dataset in 2D
    silhouettes = []
    randScores = []

    for i in range(len(combinationsKMeans)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsKMeans)))
        labels = kmeans(umap2d, combinationsKMeans[i][0], combinationsKMeans[i][1])
        silhouetteKNN = silhouette(umap2d, labels); silhouettes.append(silhouetteKNN)
        randKNN = rand(target, labels); randScores.append(randKNN)
        imgName = 'KMeans'+str(combinationsKMeans[i][0])
        plot2dClusters(umap2d, labels, imgName, combinationsKMeans[i][1])
    saveResultFileKmeans(combinationsKMeans, 'UMAP', silhouettes, randScores)

    print('Higher Dimensions') #Repeticion of the same procedure for dataset in higher dimensions
    silhouettes = []
    randScores = []

    for i in range(len(combinationsKMeans)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsKMeans)))
        labels = kmeans(higher_dim, combinationsKMeans[i][0], combinationsKMeans[i][1])
        silhouetteKNN = silhouette(higher_dim, labels); silhouettes.append(silhouetteKNN)
        randKNN = rand(target, labels); randScores.append(randKNN)
    saveResultFileKmeans(combinationsKMeans, 'HigherDims', silhouettes, randScores)


    print('Evaluating clustering algorithms: Fuzzy C Means and Probabilistic C Means')
    # Parameters for the Fuzzy C Means and Probabilistic C Means algorithms
    clusters = [ogClusters-1,ogClusters,ogClusters+1]
    fuzzifier = [1.2, 1.4, 1.6]
    error = 0.001
    maxiter = 1000

    combinationsFuzzy = list(itertools.product(clusters, fuzzifier, norms))

    print('Normal Dimensions')

    randFCM = []; silhouetteFCM = []
    randPCM = []; silhouettePCM = []
    
    for i in range(len(combinationsFuzzy)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsFuzzy)))
        centersFCM = fcm(features.T, combinationsFuzzy[i][0], combinationsFuzzy[i][1], error, maxiter, combinationsFuzzy[i][2])[0] #Applies the Fuzzy C Means algorithm
        labelsFCM = createClusters(features, centersFCM, combinationsFuzzy[i][2]) #Creates the clusters
        #Calculates the silhouette and rand scores
        silhouetteFCM.append(silhouette(features, labelsFCM))
        randFCM.append(rand(target, labelsFCM))

        centersPCM = pcm(features.T, combinationsFuzzy[i][0], combinationsFuzzy[i][1], error, maxiter, combinationsFuzzy[i][2])[0] #Applies the Probabilistic C Means algorithm
        labelsPCM = createClusters(features, centersPCM, combinationsFuzzy[i][2]) #Creates the clusters
        #Calculates the silhouette and rand scores
        silhouettePCM.append(silhouette(features, labelsPCM))
        randPCM.append(rand(target, labelsPCM))
    saveResultFileCMeans(combinationsFuzzy, 'NormalDims', silhouetteFCM, randFCM, silhouettePCM, randPCM) #Saves the results to .txt files

    print('UMAP') #Repetition of the same procedure for dataset in 2D

    randFCM = []; silhouetteFCM = []
    randPCM = []; silhouettePCM = []

    for i in range(len(combinationsFuzzy)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsFuzzy)))
        centersFCM = fcm(umap2d.T, combinationsFuzzy[i][0], combinationsFuzzy[i][1], error, maxiter, combinationsFuzzy[i][2])[0]
        labelsFCM = createClusters(umap2d, centersFCM, combinationsFuzzy[i][2])
        silhouetteFCM.append(silhouette(umap2d, labelsFCM))
        randFCM.append(rand(target, labelsFCM))
        imgName = 'Fuzzy'+str(combinationsFuzzy[i][:-1])
        plot2dClusters(umap2d, labelsFCM, imgName, combinationsFuzzy[i][2])

        centersPCM = pcm(umap2d.T, combinationsFuzzy[i][0], combinationsFuzzy[i][1], error, maxiter, combinationsFuzzy[i][2])[0]
        labelsPCM = createClusters(umap2d, centersPCM, combinationsFuzzy[i][2])
        silhouettePCM.append(silhouette(umap2d, labelsPCM))
        randPCM.append(rand(target, labelsPCM))
        imgName = 'Prob'+str(combinationsFuzzy[i][:-1])
        plot2dClusters(umap2d, labelsPCM, imgName, combinationsFuzzy[i][2])
    saveResultFileCMeans(combinationsFuzzy, 'UMAP', silhouetteFCM, randFCM, silhouettePCM, randPCM)

    print('Higher Dimensions') #Repeticion of the same procedure for dataset in higher dimensions

    randFCM = []; silhouetteFCM = []
    randPCM = [0]; silhouettePCM = [0]

    for i in range(len(combinationsFuzzy)):
        print('Currently running combination '+str(i+1)+' of '+str(len(combinationsFuzzy)))
        centersFCM = fcm(higher_dim.T, combinationsFuzzy[i][0], combinationsFuzzy[i][1], error, maxiter, combinationsFuzzy[i][2])[0]
        labelsFCM = createClusters(higher_dim, centersFCM, combinationsFuzzy[i][2])
        silhouetteFCM.append(silhouette(higher_dim, labelsFCM))
        randFCM.append(rand(target, labelsFCM))

        #centersPCM = pcm(higher_dim.T, combinationsFuzzy[i][0], combinationsFuzzy[i][1], error, maxiter, combinationsFuzzy[i][2])[0]
        #labelsPCM = createClusters(higher_dim, centersPCM, combinationsFuzzy[i][2])
        #silhouettePCM.append(silhouette(higher_dim, labelsPCM))
        #randPCM.append(rand(target, labelsPCM))
    saveResultFileCMeans(combinationsFuzzy, 'HigherDims', silhouetteFCM, randFCM, silhouettePCM, randPCM)

    print('Done')
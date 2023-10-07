'''
Paulina Pérez Garcés

main.py is the main file of the project. It imports all the necessary modules 
and runs the algorithms. It also creates a virtual environment and installs all
the necessary packages.
'''

if __name__ == "__main__":

    #Basic imports for creating the virtual environment
    from subprocess import run, Popen

    #Creation of the virtual environment
    run(["python", "-m", "venv", "venvIA"])
    Popen(["source ", "venvIA/bin/activate"],shell=True)
    run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    run(["venvIA/bin/pip", "install", "-r", "./src/requirements.txt"])

    #Import of the necessary modules from other files
    from algorithms import *
    from distances import *
    from data import *
    from autoencoder_UMAP import *

    features, target = getData() #Get and organizes the data
    higher_dim = get_MLP_encoding(features) #Gets MLP encoding to higher dims
    N, n, m = dataCharacterization(features, target) #Gets data characteristics
    norms = ['euclid', 'manhattan', 'mahalanobis', 'cosine', 'lp'] #Norms used

    #Lists to store the results of the algorithms
    cajas_df = []
    knn_df = []

    for norma in norms:
        print('currently running norm: '+norma)
        norm = applyNorm(norma, features) #Applies the norm to the data
        groups_cajas = boxes(norm, 0.2) #Applies the boxes algorithm
        groups_knn = pseudo_knn(norm, 15) #Applies the knn-like algorithm
        cajas_df.append(groups_cajas) #Stores the results
        knn_df.append(groups_knn)
    
    #Saves the results to .txt files
    saveResultFile(cajas_df, knn_df, norms, N, n, m)

    print('Done')
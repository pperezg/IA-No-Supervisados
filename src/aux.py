import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def olga2Labels(results):
    resultLabels = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            if results[i][j]==1:
                resultLabels.append(j)

    return resultLabels

def plot2dClusters(data, labels, algorithm, norm):

    sns.set()
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=labels, palette='viridis')
    sns.set_style('whitegrid')
    sns.despine()
    sns.set_context('paper')
    sns.set(font_scale=1.5)
    plt.title('Results for '+algorithm+' algorithm using the '+norm+ ' norm')
    plt.savefig('src/results/'+algorithm+'_'+norm+'.png')
    plt.clf()
    plt.close()
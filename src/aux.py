import numpy as np

def olga2Labels(results):
    resultLabels = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            if results[i][j]==1:
                resultLabels.append(j)
    
    return resultLabels
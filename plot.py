import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn
import utils

def plotFeatures(D, L, features_list, labels_list, mode):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if (i == j):
                plt.figure()
                plt.hist(D[i, L == 0], density=True, alpha=0.6, edgecolor='black', linewidth=0.5)
                plt.hist(D[i, L == 1], density=True, alpha=0.6, edgecolor='black', linewidth=0.5)
                plt.legend(labels_list)
                plt.xlabel(features_list[i])
                plt.savefig("./features_analysis/"+mode+"/"+features_list[i]+".png")
                plt.close()
            else:
                # Else use scatter plot
                plt.figure()
                plt.scatter(D[i, L == 0], D[j, L == 0], s=5)
                plt.scatter(D[i, L == 1], D[j, L == 1], s=5)
                plt.legend(labels_list)
                plt.ylabel(features_list[j])
                plt.xlabel(features_list[i])
                plt.savefig("./features_analysis/"+mode+"/"+features_list[i]+"-"+features_list[j]+".png")
                plt.close()
    return            

def heatmap(D, L, note=False):
    
    if(note == False):
        folder = "no_an"
    else:
        folder = "an"
    
    # Training set
    plt.figure()
    plt.title("Heatmap of the whole training set", pad = 12.0)
    seaborn.heatmap(abs(np.corrcoef(D)), cmap="Greys", annot=note)
    plt.savefig("./heatmaps/"+folder+"/all.png", dpi=1200)
    plt.close()
    
    # False class
    plt.figure()
    plt.title("Heatmap of the false class", pad = 12.0)
    seaborn.heatmap(abs(np.corrcoef(D[:, L==0])), cmap="Reds", annot=note)
    plt.savefig("./heatmaps/"+folder+"/false.png", dpi=1200)
    plt.close()
    
    # True class
    plt.figure()
    plt.title("Heatmap of the true class", pad = 12.0)
    seaborn.heatmap(abs(np.corrcoef(D[:, L==1])), cmap="Blues", annot=note)
    plt.savefig("./heatmaps/"+folder+"/true.png", dpi=1200)
    plt.close()

    return





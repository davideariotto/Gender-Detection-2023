import utils
import plot
import numpy as np


# Load the training data
D,L = utils.load("data/Train.txt")

print(D.shape)
print(L.shape)

# Features exploration
plot.plotFeatures(D, L, utils.features_list, utils.classes_list, "original")
normalizedData, normalizedMean, normalizedStandardDeviation = utils.ZNormalization(D)
plot.plotFeatures(normalizedData, L, utils.features_list, utils.classes_list, "normalized")

# Correlation analysis
plot.heatmap(normalizedData, L, True)
plot.heatmap(normalizedData, L, False)
    
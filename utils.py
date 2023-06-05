import numpy as np
from scipy.stats import norm

classes_list = ["Male", "Female"]
features_list = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"]

# Transform 1-dim row vectors to column vectors.
def mcol(v):
    return v.reshape((v.size, 1))

# Transform 1-dim column  vecotrs to row vectors.
def mrow(v):
    return (v.reshape(1, v.size))

# Load data
def load(filename):
    data_list = []
    labels_list = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] != '\n':
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                data[-1] = int(data[-1].rstrip('\n'))
                # Create a column vector and append it
                data_list.append(mcol(np.array(data[0:-1])))
                labels_list.append(data[-1])
    # Stack horizontally all column vectors to form a matrix
    dataset_matrix = np.hstack(data_list[:])
    # Create a 1-dim array with class labels
    labels_array = np.array(labels_list)
    
    return dataset_matrix, labels_array

# Transform the data into a distribution with 0 mean and 1 variance
def ZNormalization(D):
    mu = D.mean(axis=1)
    sigma = D.std(axis=1)
    normalizedData = (D - mcol(mu)) / mcol(sigma)
    return normalizedData, mu, sigma

# Perform gaussianization
def Gaussianization(D):
    
    TD = np.zeros(D.shape)
    
    '''
    ranks=[]
    for j in range(D.shape[0]):
        tempSum=0
        for i in range(TD.shape[1]):
            tempSum+=(D[j, :]<TD[j, i]).astype(int)
        tempSum+=1
        ranks.append(tempSum/(TD.shape[1]+2))
    y = norm.ppf(ranks)
    '''
    
    for i in range(D.shape[1]):
        x = D[:,i]
        count = np.sum(x.reshape((x.shape[0],1))>D,axis=1)
        rank = (count+1)/ (D.shape[1]+2)
        TD[:,i]= norm.ppf(rank)
     
    return TD


# Single fold
def single_fold(D, L, trainModel=None, getScores=None, train=True, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0) # size of training batch
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    # Split (using array of non overlapping indexes)
    idxTrain = idx[0:nTrain]
    idxEval = idx[nTrain:]
    # Extract training/validation data
    DTR = D[:, idxTrain]
    DEV = D[:, idxEval]
    # Extract training/validation labels
    LTR = L[idxTrain]
    LEV = L[idxEval]
    
    if (train):  
        # Train model and get the performance over the validation set
        mean0, sigma0, mean1, sigma1 = trainModel(DTR, LTR)
        scores = getScores(mean0, sigma0, mean1, sigma1, DEV)
        
        return scores, LEV
    else:
        # don't train
        return (DTR, LTR) , (DEV, LEV)
        

# K-Fold
def Kfold(D, L, trainModel=None, getScores=None, train=True, K=5, seed=0):
    
    foldSize = int(D.shape[1]/K) # size of each single fold 
    folds = []
    labels = []
    # Generate a random seed and compute a permutation from 0 to 8929
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    for i in range(K):
        fold_idx = idx[(i*foldSize) : ((i+1)*(foldSize))]
        folds.append(D[:,fold_idx])
        labels.append(L[fold_idx])
        
    scores = [] # Each iteration we save here the score for the evaluation set. At the end we have the score of the whole dataset
    evaluationLabels = [] # Each iteration we save here the labels for the evaluation set. At the end we have the labels in the new shuffled order 

    allKFolds = []
    
    # loop K times
    for i in range(K):
        # For each iteration, K-1 training folds and 1 evaluation fold 
        trainingFolds = []
        trainingLabels = []
        evaluationFold = []
       
        # Loop over the folds. The i-th one is the evaluation fold
        for j in range(K):
            if j==i:
                evaluationFold.append(folds[i])
                evaluationLabels.append(labels[i])
            else:
                trainingFolds.append(folds[j])
                trainingLabels.append(labels[j])

        trainingFolds=np.hstack(trainingFolds)
        trainingLabels=np.hstack(trainingLabels)
        evaluationFold = np.hstack(evaluationFold)
        
        if (train):
            # Train the model and calculate the score on the evaluation set
            mean0, sigma0, mean1, sigma1 = trainModel(trainingFolds, trainingLabels)
            scores.append(getScores(mean0, sigma0, mean1, sigma1, evaluationFold))
        else:
            allKFolds.append([trainingLabels, trainingFolds, evaluationFold])
    
    if(train):
        scores = np.hstack(scores)
    evaluationLabels=np.hstack(evaluationLabels)
    
    if (train):    
        return scores, evaluationLabels
    else:
        return allKFolds, evaluationLabels




"""

DIMENSIONALITY REDUCTION

"""

def PCA(D, L, m):
    
    C = covarianceMatrix(D) # covariance matrix
    
    s, U = eigh(C) # eigenvalues and eigenvectors

    P = U[:, 0:m] # principal component

    DP = np.dot(P.T, D) # PCA projection matrix
    
    return DP

def covarianceMatrix(D):
    # Covariance matrix:
    # C = 1 / N * ((X - mu) * (X - mu)^T)
    
    mu = mcol(D.mean(axis=1)) # means over the columns
    DC = D - mu     # center the data
    
    # Covariance matrix
    C = (1/DC.shape[1]) * (np.dot(DC, DC.T))
    #C = np.cov(DC, rowvar=False)
    
    return C

def eigh(matrix):
    s, U = np.linalg.eigh(matrix) # eigh for a simmetric matrix (eig for non simmetric)
    U = U[:, ::-1] # Descending order
    
    return s, U


"""

METRICS

"""

def minimum_detection_costs (llr, y, pi1, cfn, cfp):
    
    """
    llr: log-likelihood ratios.
    y: true labels of the data points in llr.
    pi1: prior probability of the positive class.
    cfn: cost of false negative.
    cfp: cost of false positive.
    """
    
    llrSorted = np.sort(llr)
    normDCF= [] # array to store the costs
    
    for t in llrSorted: # for each threshold value
        
        # Calculate the predictions (0 or 1) by comparing the log-likelihood ratios with the threshold value       
        # if llr > threshold => 1 else 0
        predictions = (llr > t).astype(int)
        
        # Confusion matrix
        CM = confusionMatrix(predictions, y, 2) # 2 = y.max()+1
        
        # Un-normalized detection cost
        DCFu = compute_dc(CM, pi1, cfn, cfp)
        
        # Normalize and append
        normDCF.append(normalized_dc(DCFu, pi1, cfn, cfp))
        
    i = np.argmin(normDCF) # get the index of the minimum cost
        
    return normDCF[i]

def confusionMatrix(predictedLabels, actualLabels, K):
    # The confusion matrix "counts" how many times we get 
    # prediction i when the actual label is j.
    matrix = np.zeros((K, K)).astype(int)
    
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    
    return matrix

def compute_dc (CM, pi1, cfn, cfp):
    """
    CM: confusion matrix, where the rows represent the actual values and the columns represent the predicted values.
    pi1: prior probability of the positive class.
    cfn: cost of false negative.
    cfp: cost of false positive.
    """
    #Calculate false negative rate (FNR) 
    FNR = CM[0][1]/(CM[0][1]+CM[1][1]) 
    #Calculate false positive rate (FPR)
    FPR = CM[1][0]/(CM[0][0]+CM[1][0])
    
    #Calculate un-normalized detection cost
    return (pi1*cfn*FNR + (1-pi1)*cfp*FPR)


def normalized_dc (DCF, pi1, cfn, cfp):
    # vector with dummy costs (dummy = system that always predict one class)
    temp = np.array([pi1*cfn, (1-pi1)*cfp]) 
    i = np.argmin(temp) # index of the optimal dummy system
    
    return DCF/temp[i] # normalized DCF
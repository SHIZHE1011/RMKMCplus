# Author: Shizhe Liang
# Last modified: 9/28/20
# This algorithm is based on Cai, Nie and Huang's paper "Multi-view K-Means Clustering on Big Data (2013)".

import numpy as np

'''Matrix 2,1-norm'''
def norm21(X):
    '''
    Parameters
    ----------
    X: 2darray of dimension (n_samples, n_features).

    Returns
    -------
    res = matrix 2,1-norm defined by Sigma_i ||xi||_2,
        where xi's are the row vectors of X.

    '''
    res = 0
    for i in range(X.shape[0]):
        res += np.linalg.norm(X[i])
    return res

'''1-hot encoding'''
def onehot(labels, k):
    '''
    Parameters
    ----------
    k: the expected number of clusters.

    labels: 1darray of cluster labels, each entry is a interger in [0, k-1].

    Returns
    -------
    G: indicator matrix of dimension (n_samples, k)
        s.t. G(i,j) == 1 iff labels[i] == j.
    '''
    n_samples = len(labels)
    G = np.zeros((n_samples, k))
    for i in range(len(labels)):
         G[i, labels[i]] = 1
    return G

'''decode 1-hot encoding'''
def deonehot(G):
    '''
    Parameters
    ----------
    G: indicator matrix of dimention (n_samples, k)

    Returns
    -------
    labels: 1darray of cluster labels,
        where labels[i] == j iff G(i,j) == 1
    '''
    n_samples = G.shape[0]
    labels = [0] * n_samples
    for i in range(n_samples):
        labels[i] = np.nonzero(G[i])[0][0]
    return labels

'''kmeans ++ initialization method'''
def kmeans_pp(X, k):
    '''
    Parameters
    -----------
    X: data matrix with rows being data points
    k: number of clusters
    
    Returns
    --------
    centroids = centroid matrix give by kmeans++ initialization
    '''
    n_samples = X.shape[0]
    first_cent = np.random.choice(n_samples, 1)
    centroids = [X[first_cent]]
    for i in range(1, k):
        min_dist = [min([np.linalg.norm(x-cent)**2 for cent in centroids]) for x in X]
        next_cent = np.random.choice(n_samples, 1, min_dist)
        centroids.append(X[next_cent])
    return np.array(centroids)

'''RMKMC Algorithm'''
def RMKMC(Xs, k, gamma, n_iter = 300, initialization = "rand"):
    '''
    Parameters
    ----------
    Xs: a list of matrices. For each of them, the dimention is (n_samples, n_features),
        so every row correpsonds to a data point.

    k: the expected number of clusters.

    gamma: the parameter controling the weights. It needs to be strictly larger than 1.

    n_iter: maximum number of iterations, default is 300.

    Returns
    -------
    G: common indicator matrix of dimension (n_samples, k).

    Fs: a list of cluster centroids matrices, each of dimention (k, n_features).

    aa: 1darray of weights for the views.
    '''

    # Security check for Xs being empty, k <= 1 or gamma == 1.
    n_views = len(Xs)
    n_samples = Xs[0].shape[0]

    if n_views == 0:
        print("No data")
        return
    if k <= 1:
        print("k less than 1")
        return
    if gamma == 1:
        print("gamma cannot be 1")
        return

    # intializations
    G = None
    Ds = [np.eye(n_samples) for _ in range(n_views)]

    Fs = [None for _ in range(n_views)]

    aa = [1/n_views for _ in range(n_views)]
    
    if initialization == 'rand':
        labels = np.random.randint(0, k, size = n_samples)
        G = onehot(labels, k)
    elif initialization == 'pp':
        # Do kmeans++ for each view and store the results into Fs
        for v in range(n_views):
            Fs[v] = kmeans_pp(Xs[v], k)

        # Update G by finding the best label for each data point
        labels = []
        for j in range(n_samples):
            # We to a brute force search
            cur_min = float("inf")
            cur_ind = 0
            for m in range(k):
                cur_sum = sum([np.linalg.norm(Xs[v][j] - Fs[v][m])**2 for v in range(n_views)])
                if cur_sum <= cur_min:
                    cur_min = cur_sum
                    cur_ind = m
            labels.append(cur_ind)
        G = onehot(labels, k)
        
    else:
        print("unknown initialzation")
        return

    # iterations
    for i in range(n_iter):
        print("loop", i)

        # Calculate tildeD for each view
        tildeDs = [(aa[v]**gamma) * Ds[v] for v in range(n_views)]

        # Update the centroids matrix F for each view.
        for v in range(n_views):
            Ftrans = Xs[v].T @ tildeDs[v] @ G @ np.linalg.inv(G.T @ tildeDs[v] @ G)
            Fs[v] = Ftrans.T

        # Update G by finding the best label for each data point
        labels = []
        for j in range(n_samples):
            # We to a brute force search
            cur_min = float("inf")
            cur_ind = 0
            for m in range(k):
                cur_sum = sum([tildeDs[v][j,j] * np.linalg.norm(Xs[v][j] - Fs[v][m])**2 for v in range(n_views)])
                if cur_sum <= cur_min:
                    cur_min = cur_sum
                    cur_ind = m
            labels.append(cur_ind)
        G = onehot(labels, k)

        # Update D for each view
        for v in range(n_views):
            for j in range(n_samples):
                Ds[v][j,j] = 1/(2 * np.linalg.norm((Xs[v] - G @ Fs[v])[j]))

        # Update weights aa
        numerator = [(gamma * norm21(Xs[v] - G @ Fs[v]))**(1/(1-gamma)) for v in range(n_views)]
        denominator = sum(numerator)

        for v in range(n_views):
            aa[v] = numerator[v] / denominator
            # Security check
            if np.isnan(aa[v]):
                print("linalg error")
                return

    return (G, Fs, aa)
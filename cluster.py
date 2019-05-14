import numpy as np
import sys

from sklearn.cluster import KMeans
from utils import read_models

# number of cluster centers
k = 3 if len(sys.argv) != 2 else int(sys.argv[1])

# models are parameterized by the matrices X and Y
models = [(model.X, model.Y, file) for model, file in read_models('.', '*.model')]
files = [model[-1] for model in models]
models = [(X, Y) for X, Y, _ in models]

# model parameters
N, d = len(models), models[0][0].shape[0]

# compute mean and stdev, used for normalization
def compute_stats(models):
	X_mean, Y_mean = sum(X for X, _ in models), sum(Y for _, Y in models)
	X_mean, Y_mean = 1/N * X_mean, 1/N * Y_mean 

	X_std = np.std(tuple(X for X, _ in models), axis=0, ddof=1)
	Y_std = np.std(tuple(Y for _, Y in models), axis=0, ddof=1)

	# avoid divide by zero
	X_std[X_std == 0.0] = 1e-8
	Y_std[Y_std == 0.0] = 1e-8

	return (X_mean, Y_mean), (X_std, Y_std)

# subtract mean and divide by stdev
def normalize(model, X_mean, Y_mean, X_std, Y_std):
	X, Y = model
	return ((X - X_mean)/X_std, (Y - Y_mean)/Y_std)

# un-normalize
def inverse_normalize(model, X_mean, Y_mean, X_std, Y_std):
	X, Y = model
	return (X*X_std + X_mean, Y*Y_std + Y_mean)


(X_mean, Y_mean), (X_std, Y_std) = compute_stats(models)
models = [normalize(model, X_mean, Y_mean, X_std, Y_std) for model in models]

# flatten parameters into vector, then aggregate into cluster matrix 
vectors = [np.concatenate((X.reshape((1,-1)), Y.reshape((1,-1))), axis=1) for X, Y in models]
vectors = np.concatenate(tuple(vectors))

# compute and report clusters
kmeans = KMeans(n_clusters=k).fit(vectors)
print('{}\t\t{}'.format('Filename', 'Cluster Assignment'))
for i, file in enumerate(files):
	print('{}\t{}'.format(file ,kmeans.labels_[i]))
import numpy as np

def minkowski_dist(X, Y, p=1):
  return (np.sum((abs(X - Y)**p))) ** (1/p)

def get_dist_matrix(data):
  return np.asarray([list(map(lambda X: minkowski_dist(X, point), data)) for point in data])

def get_max_r_kmeans(sk_centers, data):
  dists = np.asarray([list(map(lambda X: minkowski_dist(X, point), data)) for point in sk_centers])
  return np.amax(dists)
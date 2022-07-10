import numpy as np

def minkowski_dist(X, Y, p=1):
  return (np.sum((abs(X - Y)**p))) ** (1/p)

def get_dist_matrix(data):
  return np.asarray([list(map(lambda X: minkowski_dist(X, point), data)) for point in data])
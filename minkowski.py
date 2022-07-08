import numpy as np

def minkowski_dist(X, Y, p=1):
  return (np.sum((abs(X - Y))**p)) ** (1/p)

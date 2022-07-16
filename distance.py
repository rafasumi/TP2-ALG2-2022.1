import numpy as np

# Computa a distância de Minkowski entre X e Y, dado o parâmetro p
def minkowski_dist(X, Y, p=1):
  return (np.sum((abs(X - Y)**p))) ** (1/p)

# Computa a matriz de distância para um conjunto de dados 'data' usando a distância de Minkowski com p=1
def get_dist_matrix(data):
  return np.asarray([list(map(lambda X: minkowski_dist(X, point), data)) for point in data])

# Computa o raio máximo para os k centros retornados pelo k-means do scikit-learn
def get_max_r_kmeans(sk_centers, data):
  dists = np.asarray([list(map(lambda X: minkowski_dist(X, point), data)) for point in sk_centers])
  return np.amax(dists)
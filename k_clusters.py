import numpy as np

# Implementação do algoritmo de k_clusters
def k_clusters(dist_matrix, n, k):
  if k >= n:
    return (np.arange(n), 0)

  random_index = np.random.randint(0, n)
  # Lista com todos os pontos que não são centros
  points = list(range(n))

  # Listas com os centros obtidos até o momento
  centers = [random_index]
  points.pop(random_index)

  while len(centers) < k:
    # Encontra um ponto que não é centro e que está a uma distância máxima de um ponto que é centro
    max_dist_index = np.unravel_index(np.argmax(dist_matrix[np.ix_(centers, points)]), (len(centers), len(points)))[1]
    # Remove da lista de pontos e adiciona na lista de centros
    index = points.pop(max_dist_index)
    centers.append(index)

  max_r = np.amax(dist_matrix[np.ix_(centers, points)])
  labels = np.argmin(dist_matrix[centers, :], axis=0)

  return (labels, max_r)

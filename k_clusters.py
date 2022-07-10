import numpy as np

def k_clusters(dist_matrix, n, k):
  if k >= n:
    return (np.zeros(n), 0)

  rand_index = np.random.randint(0, n)
  points = list(range(n))

  centers = [rand_index]
  points.pop(rand_index)

  while len(centers) < k:
    max_dist_index = np.unravel_index(np.argmax(dist_matrix[np.ix_(centers, points)]), (len(centers), len(points)))[1]
    index = points.pop(max_dist_index)
    centers.append(index)
  
  max_r = np.amax(dist_matrix[np.ix_(centers, points)])
  labels = np.argmin(dist_matrix[centers, :], axis=0)

  return (labels, max_r)

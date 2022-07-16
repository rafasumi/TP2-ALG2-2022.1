import numpy as np
import pandas as pd
from optparse import OptionParser
from sklearn.cluster import k_means
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from time import time
from os.path import exists
from distance import get_dist_matrix, get_max_r_kmeans
from k_clusters import k_clusters

def main():
  parser = OptionParser()
  parser.add_option('-i', '--input', action='store', help='Diretório do arquivo de input', type='string', dest='input_file')
  parser.add_option('-o', '--output', action='store', help='Nome dos arquivos CSV de output (sem a extensão .csv)', type='string', default='results', dest='output_file')
  parser.add_option('--iters', action='store', help='Número de execuções do algoritmo a serem feitas', type='int', default=30, dest='iters')
  parser.add_option('--seed', action='store', help='Valor para a seed do numpy.random', type='int', default=42, dest='seed')

  (options, _) = parser.parse_args()

  input_file = options.input_file
  if not input_file or not exists(input_file):
    raise Exception(f'O arquivo de input informado é inválido. Tente usar a opção --help para mais informações')

  np.random.seed(options.seed)

  # Lendo os dados do arquivo CSV
  data = pd.read_csv(input_file, sep=',')
  # Removendo linhas com valores nulos
  data = data.dropna()
  # Faz um encoding da coluna de target, caso não seja numérica
  if pd.api.types.is_string_dtype(data.iloc[:, -1].dtype):
    data.iloc[:, -1] = data.iloc[:, -1].astype('category').cat.codes
  # Escolhendo valor de k com base no número de classes
  k = data.iloc[:, -1].nunique()

  # Convertendo o DataFrame para um array numpy n-dimensional
  data = data.to_numpy()

  target = data[:, -1]
  data = data[:, :-1]

  # Normalização dos dados
  mean = data.mean()
  std = data.std()
  data = (data - mean)/std

  dist_matrix = get_dist_matrix(data)

  results = []
  for _ in range(options.iters):
    start_time = time()
    labels, max_r = k_clusters(dist_matrix, data.shape[0], k)
    execution_time = time() - start_time
    max_r = (max_r * std) + mean
    results.append([
      adjusted_rand_score(target, labels),
      silhouette_score(data, labels),
      round(max_r, 7),
      round(execution_time, 7)
    ])
  results_df = pd.DataFrame(results, columns=['Rand ajustado', 'Silhueta', 'Raio máximo', 'Tempo de execução']).describe()
  results_df.to_csv(f'./out/{options.output_file}.csv')
  
  results = []
  for i in range(options.iters):
    start_time = time()
    sk_centers, sk_labels, _ = k_means(X=data, n_clusters=k, max_iter=1, n_init=1, random_state=i)
    execution_time = time() - start_time
    max_r_kmeans = get_max_r_kmeans(sk_centers, data)
    max_r_kmeans = (max_r_kmeans * std) + mean
    results.append([
      adjusted_rand_score(target, sk_labels),
      silhouette_score(data, sk_labels),
      round(max_r_kmeans, 7),
      round(execution_time, 7)
    ])
  results_sk_learn_df = pd.DataFrame(results, columns=['Rand ajustado', 'Silhueta', 'Raio máximo', 'Tempo de execução']).describe()
  results_sk_learn_df.to_csv(f'./out/{options.output_file}_sklearn.csv')

if __name__ == '__main__':
  main()

import numpy as np
import pandas as pd
from optparse import OptionParser
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from time import time
from os.path import exists
from minkowski import get_dist_matrix
from k_clusters import k_clusters

def main():
  parser = OptionParser()
  parser.add_option('-i', '--input', action='store', help='Diretório do arquivo de input', type='string', dest='input_file')
  parser.add_option('-o', '--output', action='store', help='Nome do arquivo CSV de output', type='string', default='results.csv', dest='output_file')
  parser.add_option('-n', action='store', help='Número de execuções do algoritmo a serem feitas', type='int', default=30, dest='iters')
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

  # Removendo linhas duplicadas
  data = data.drop_duplicates()

  # Faz um encoding da coluna de target caso não seja numérica
  if pd.api.types.is_string_dtype(data.iloc[:, -1].dtype):
    data.iloc[:, -1] = data.iloc[:, -1].astype('category').cat.codes
  k = data.iloc[:, -1].nunique()

  # Convertendo o DataFrame para um array numpy n-dimensional
  data = data.to_numpy()

  target = data[:, -1]
  data = data[:, :-1]
  # TO-DO: NORMALIZAR (e depois desnormalizar o max_r)
  # TO-DO: EXECUTAR KMEANS
  dist_matrix = get_dist_matrix(data)
  results = []
  for _ in range(30):
    start_time = time()
    labels, max_r = k_clusters(dist_matrix, data.shape[0], k)
    execution_time = time() - start_time
    results.append([
      adjusted_rand_score(target, labels),
      silhouette_score(data, labels),
      round(max_r, 7),
      round(execution_time, 7)
    ])
  
  results_df = pd.DataFrame(results, columns=['Rand ajustado', 'Silhueta', 'Raio máximo', 'Tempo de execução']).describe()
  results_df.to_csv(f'./out/{options.output_file}')

if __name__ == '__main__':
  main()

import numpy as np
import pandas as pd
from optparse import OptionParser
from os.path import exists
from minkowski import get_dist_matrix
from k_clusters import k_clusters

def main():
  parser = OptionParser()
  parser.add_option('-i', '--input', action='store', help='Arquivo de input', type='string', dest='input_file')
  parser.add_option('-k', action='store', help='Valor de k para o algoritmo de k-clusters', type='int', default=8, dest='k')
  parser.add_option('-n', action='store', help='Número de execuções do algoritmo a serem feitas', type='int', default=30, dest='iters')

  (options, _) = parser.parse_args()

  input_file = options.input_file
  if not input_file or not exists(input_file):
    raise Exception(f'O arquivo de input informado é inválido. Tente usar a opção --help para mais informações')
  
  # Lendo os dados do arquivo CSV
  data = pd.read_csv(input_file, sep=',')
  # Removendo linhas com valores nulos
  data = data.dropna()
  # Removendo linhas duplicadas
  data = data.drop_duplicates()
  # Convertendo o DataFrame para um array numpy n-dimensional
  data = data.to_numpy()

  target = data[:, -1]
  data = data[:, :-1]

  dist_matrix = get_dist_matrix(data)
  for i in range(options.iters):
    np.random.seed(i)
    k_clusters(dist_matrix, data.shape[1], options.k)
    

if __name__ == '__main__':
  main()

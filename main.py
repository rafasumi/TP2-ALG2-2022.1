import pandas as pd
from optparse import OptionParser
from os.path import exists

def main():
  parser = OptionParser()
  parser.add_option('-i', action='store', type='string', dest='input_file')

  (options, _) = parser.parse_args()

  input_file = options.input_file
  if not exists(input_file):
    raise Exception(f'O arquivo "{input_file}" não existe')
  
  data = pd.read_csv(input_file, sep=',').dropna()._get_numeric_data().to_numpy()

  print(data)

if __name__ == '__main__':
  main()

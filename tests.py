from subprocess import run

input_files = [
  'audit_risk',
  'cmc',
  'data_banknote_authentication',
  'mammographic_masses',
  'Maternal_Health_Risk_Data_Set',
  'ObesityDataSet',
  'Raisin_Dataset',
  'Shill_Bidding_Dataset',
  'transfusion',
  'yeast',
  'iris'
]

if __name__ == '__main__':
  for input in input_files:
    run(['python3', './main.py', '-i', f'./data/{input}.csv', '-o', f'results_{input}'])
    print(f'{input} executado')
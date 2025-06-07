import pandas as pd

try:
    df = pd.read_csv('respostas.csv')
    print("\n--- As colunas no seu arquivo CSV são: ---")
    for col_name in df.columns.tolist():
        print(f"- {col_name}")
    print("-------------------------------------------\n")
except FileNotFoundError:
    print("Erro: O arquivo 'respostas.csv' não foi encontrado. Certifique-se de que ele está na mesma pasta do script.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
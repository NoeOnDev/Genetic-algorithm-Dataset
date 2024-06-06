import pandas as pd

# Aquí cargo el dataset
dataset = pd.read_excel('dataset.xlsx')

# Aquí almaceno las variables de entrada para cada fila
x1 = dataset['x1'].tolist() # Convierto la columna x1 en una lista 31 elementos
x2 = dataset['x2'].tolist() # Convierto la columna x2 en una lista 31 elementos
x3 = dataset['x3'].tolist() # Convierto la columna x3 en una lista 31 elementos
x4 = dataset['x4'].tolist() # Convierto la columna x4 en una lista 31 elementos

# Aquí almaceno la "y" deseada para cada fila
yd = dataset['y'].tolist()


import pandas as pd

# Aquí cargo el dataset
dataset = pd.read_excel('dataset.xlsx')

# Aquí almaceno las variables de entrada para cada fila
x1 = dataset['x1'].tolist()
x2 = dataset['x2'].tolist()
x3 = dataset['x3'].tolist()
x4 = dataset['x4'].tolist()
# Aquí almaceno la "y" deseada para cada fila
yd = dataset['y'].tolist()

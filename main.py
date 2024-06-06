import pandas as pd

dataset = pd.read_excel('dataset.xlsx')

x1 = dataset['x1'].tolist()
x2 = dataset['x2'].tolist()
x3 = dataset['x3'].tolist()
x4 = dataset['x4'].tolist()
y = dataset['y'].tolist()

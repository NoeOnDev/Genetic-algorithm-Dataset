# archivo: interfaz_algoritmo.py

import tkinter as tk
from tkinter import ttk
from algoritmo_genetico import ejecutar_algoritmo_genetico

def ejecutar_algoritmo():
    tam_poblacion = int(tam_poblacion_var.get())
    tam_max_poblacion = int(tam_max_poblacion_var.get())
    x_min = float(x_min_var.get())
    x_max = float(x_max_var.get())
    tasa_mutacion_individuo = float(tasa_mutacion_individuo_var.get())
    tasa_mutacion_gen = float(tasa_mutacion_gen_var.get())
    generaciones = int(generaciones_var.get())
    maximizar = maximizar_var.get() == "Maximizar"
    directorio_graficas = directorio_graficas_var.get()
    directorio_evolucion = directorio_evolucion_var.get()
    output_video = output_video_var.get()
    
    ejecutar_algoritmo_genetico(tam_poblacion, tam_max_poblacion, x_min, x_max, tasa_mutacion_individuo, tasa_mutacion_gen, generaciones, maximizar, directorio_graficas, directorio_evolucion, output_video)

# Crear la ventana principal
root = tk.Tk()
root.title("Interfaz Algoritmo Genético")

# Variables
tam_poblacion_var = tk.StringVar(value='3')
tam_max_poblacion_var = tk.StringVar(value='50')
x_min_var = tk.StringVar(value='-10')
x_max_var = tk.StringVar(value='40')
tasa_mutacion_individuo_var = tk.StringVar(value='0.7')
tasa_mutacion_gen_var = tk.StringVar(value='0.6')
generaciones_var = tk.StringVar(value='100')
maximizar_var = tk.StringVar(value='Maximizar')
directorio_graficas_var = tk.StringVar(value='graficas')
directorio_evolucion_var = tk.StringVar(value='evolucion')
output_video_var = tk.StringVar(value='evolucion.mp4')

# Crear los widgets
tk.Label(root, text="Tamaño de la Población Inicial:").grid(row=0, column=0)
tk.Entry(root, textvariable=tam_poblacion_var).grid(row=0, column=1)

tk.Label(root, text="Tamaño Máximo de la Población:").grid(row=1, column=0)
tk.Entry(root, textvariable=tam_max_poblacion_var).grid(row=1, column=1)

tk.Label(root, text="Límite Inferior de x:").grid(row=2, column=0)
tk.Entry(root, textvariable=x_min_var).grid(row=2, column=1)

tk.Label(root, text="Límite Superior de x:").grid(row=3, column=0)
tk.Entry(root, textvariable=x_max_var).grid(row=3, column=1)

tk.Label(root, text="Tasa de Mutación del Individuo:").grid(row=4, column=0)
tk.Entry(root, textvariable=tasa_mutacion_individuo_var).grid(row=4, column=1)

tk.Label(root, text="Tasa de Mutación del Gen:").grid(row=5, column=0)
tk.Entry(root, textvariable=tasa_mutacion_gen_var).grid(row=5, column=1)

tk.Label(root, text="Número de Generaciones:").grid(row=6, column=0)
tk.Entry(root, textvariable=generaciones_var).grid(row=6, column=1)

tk.Label(root, text="Maximizar o Minimizar:").grid(row=7, column=0)
ttk.Combobox(root, textvariable=maximizar_var, values=["Maximizar", "Minimizar"]).grid(row=7, column=1)

tk.Label(root, text="Directorio para Guardar las Gráficas:").grid(row=8, column=0)
tk.Entry(root, textvariable=directorio_graficas_var).grid(row=8, column=1)

tk.Label(root, text="Directorio para Guardar la Evolución:").grid(row=9, column=0)
tk.Entry(root, textvariable=directorio_evolucion_var).grid(row=9, column=1)

tk.Label(root, text="Nombre del Archivo de Salida del Video:").grid(row=10, column=0)
tk.Entry(root, textvariable=output_video_var).grid(row=10, column=1)

tk.Button(root, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).grid(row=11, columnspan=2)

# Iniciar el bucle de eventos
root.mainloop()

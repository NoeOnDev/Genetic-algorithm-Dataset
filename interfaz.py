import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from algoritmo_genetico import algoritmo_genetico, binario_a_decimal, crear_video_de_imagenes

def ejecutar_algoritmo():
    try:
        tamano_poblacion = int(entrada_tamano_poblacion.get())
        tamano_maximo_poblacion = int(entrada_tamano_maximo_poblacion.get())
        generaciones = int(entrada_generaciones.get())
        prob_mutacion_individuo = float(entrada_prob_mutacion_individuo.get())
        prob_mutacion_gen = float(entrada_prob_mutacion_gen.get())
        x_min = float(entrada_x_min.get())
        x_max = float(entrada_x_max.get())
        proporcion_cromosoma = float(entrada_longitud_cromosoma.get())
        maximizar = var_maximizar.get() == 1

        poblacion_final, aptitud_final = algoritmo_genetico(tamano_poblacion, tamano_maximo_poblacion, generaciones, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, proporcion_cromosoma, maximizar)

        indice_mejor = np.argmax(aptitud_final) if maximizar else np.argmin(aptitud_final)
        mejor_solucion = binario_a_decimal(poblacion_final[indice_mejor], x_min, x_max)
        etiqueta_resultado.config(text=f'Mejor solución: x = {mejor_solucion}, f(x) = {aptitud_final[indice_mejor]}')

        crear_video_de_imagenes('graficas_generacion', 'graficas_estadisticas/evolucion_generacion.avi')
        
        # Cargar los resultados del mejor individuo
        mejores_individuos_df = pd.read_csv('mejores_individuos.csv')

        # Mostrar la tabla en una ventana de la interfaz gráfica
        mostrar_tabla_resultados(mejores_individuos_df)

    except ValueError as e:
        messagebox.showerror("Error de entrada", f"Por favor, ingrese valores válidos.\n\nDetalles del error: {e}")

def mostrar_tabla_resultados(resultados_df):
    ventana_resultados = tk.Toplevel(root)
    ventana_resultados.title("Resultados")

    tabla = ttk.Treeview(ventana_resultados, columns=('Generación', 'Individuo', 'Valor del índice', 'Valor de x', 'Aptitud'), show='headings')
    tabla.heading('Generación', text='Generación')
    tabla.heading('Individuo', text='Individuo')
    tabla.heading('Valor del índice', text='Valor del índice')
    tabla.heading('Valor de x', text='Valor de x')
    tabla.heading('Aptitud', text='Aptitud')

    for _, fila in resultados_df.iterrows():
        tabla.insert('', 'end', values=(fila['Generación'], fila['Individuo'], fila['Valor del índice'], fila['Valor de x'], fila['Aptitud']))

    tabla.pack(expand=True, fill='both')

root = tk.Tk()
root.title("Algoritmo Genético")

# Parámetros de entrada
ttk.Label(root, text="Tamaño de la población inicial:").grid(row=0, column=0, padx=10, pady=5)
entrada_tamano_poblacion = ttk.Entry(root)
entrada_tamano_poblacion.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Tamaño máximo de la población:").grid(row=1, column=0, padx=10, pady=5)
entrada_tamano_maximo_poblacion = ttk.Entry(root)
entrada_tamano_maximo_poblacion.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text="Número de generaciones:").grid(row=2, column=0, padx=10, pady=5)
entrada_generaciones = ttk.Entry(root)
entrada_generaciones.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(root, text="Probabilidad de mutación del individuo:").grid(row=3, column=0, padx=10, pady=5)
entrada_prob_mutacion_individuo = ttk.Entry(root)
entrada_prob_mutacion_individuo.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(root, text="Probabilidad de mutación del gen:").grid(row=4, column=0, padx=10, pady=5)
entrada_prob_mutacion_gen = ttk.Entry(root)
entrada_prob_mutacion_gen.grid(row=4, column=1, padx=10, pady=5)

ttk.Label(root, text="Valor mínimo de x:").grid(row=5, column=0, padx=10, pady=5)
entrada_x_min = ttk.Entry(root)
entrada_x_min.grid(row=5, column=1, padx=10, pady=5)

ttk.Label(root, text="Valor máximo de x:").grid(row=6, column=0, padx=10, pady=5)
entrada_x_max = ttk.Entry(root)
entrada_x_max.grid(row=6, column=1, padx=10, pady=5)

ttk.Label(root, text="Longitud del cromosoma (proporción):").grid(row=7, column=0, padx=10, pady=5)
entrada_longitud_cromosoma = ttk.Entry(root)
entrada_longitud_cromosoma.grid(row=7, column=1, padx=10, pady=5)

ttk.Label(root, text="Maximización:").grid(row=8, column=0, padx=10, pady=5)
var_maximizar = tk.IntVar()
ttk.Checkbutton(root, variable=var_maximizar).grid(row=8, column=1, padx=10, pady=5)

ttk.Button(root, text="Ejecutar", command=ejecutar_algoritmo).grid(row=9, column=0, columnspan=2, pady=10)

etiqueta_resultado = ttk.Label(root, text="")
etiqueta_resultado.grid(row=10, column=0, columnspan=2, pady=10)

root.mainloop()

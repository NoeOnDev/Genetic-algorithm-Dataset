import tkinter as tk
from tkinter import ttk
from algoritmo import AlgoritmoGeneticoDTO, algoritmo_genetico, funcion_fitness

def ejecutar_algoritmo():
    tam_poblacion = int(entry_tam_poblacion.get())
    max_poblacion = int(entry_max_poblacion.get())
    prob_mutacion_individuo = float(entry_prob_mutacion_individuo.get())
    prob_mutacion_gen = float(entry_prob_mutacion_gen.get())
    x_min = float(entry_x_min.get())
    x_max = float(entry_x_max.get())
    num_generaciones = int(entry_num_generaciones.get())
    modo = modo_var.get()

    dto = AlgoritmoGeneticoDTO(tam_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, num_generaciones, modo)
    mejor_solucion = algoritmo_genetico(dto)
    resultado.set(f"La mejor solución es: {mejor_solucion:.2f} con valor: {funcion_fitness(mejor_solucion):.2f}")

root = tk.Tk()
root.title("Algoritmo Genético")

ttk.Label(root, text="Tamaño de Población:").grid(row=0, column=0)
entry_tam_poblacion = ttk.Entry(root)
entry_tam_poblacion.grid(row=0, column=1)

ttk.Label(root, text="Máximo de Población:").grid(row=1, column=0)
entry_max_poblacion = ttk.Entry(root)
entry_max_poblacion.grid(row=1, column=1)

ttk.Label(root, text="Probabilidad de Mutación (Individuo):").grid(row=2, column=0)
entry_prob_mutacion_individuo = ttk.Entry(root)
entry_prob_mutacion_individuo.grid(row=2, column=1)

ttk.Label(root, text="Probabilidad de Mutación (Gen):").grid(row=3, column=0)
entry_prob_mutacion_gen = ttk.Entry(root)
entry_prob_mutacion_gen.grid(row=3, column=1)

ttk.Label(root, text="Valor mínimo de x:").grid(row=4, column=0)
entry_x_min = ttk.Entry(root)
entry_x_min.grid(row=4, column=1)

ttk.Label(root, text="Valor máximo de x:").grid(row=5, column=0)
entry_x_max = ttk.Entry(root)
entry_x_max.grid(row=5, column=1)

ttk.Label(root, text="Número de Generaciones:").grid(row=6, column=0)
entry_num_generaciones = ttk.Entry(root)
entry_num_generaciones.grid(row=6, column=1)

modo_var = tk.StringVar()
modo_var.set('maximizar')
ttk.Label(root, text="Modo:").grid(row=7, column=0)
ttk.Radiobutton(root, text='Maximizar', variable=modo_var, value='maximizar').grid(row=7, column=1)
ttk.Radiobutton(root, text='Minimizar', variable=modo_var, value='minimizar').grid(row=7, column=2)

ttk.Button(root, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).grid(row=8, column=0, columnspan=3)

resultado = tk.StringVar()
ttk.Label(root, textvariable=resultado).grid(row=9, column=0, columnspan=3)

root.mainloop()

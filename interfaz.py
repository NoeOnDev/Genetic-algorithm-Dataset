import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from algoritmo_genetico import genetic_algorithm, binary_to_decimal, create_video_from_images

def run_algorithm():
    try:
        population_size = int(entry_population_size.get())
        generations = int(entry_generations.get())
        mutation_prob_individual = float(entry_mutation_prob_individual.get())
        mutation_prob_gene = float(entry_mutation_prob_gene.get())
        x_min = float(entry_x_min.get())
        x_max = float(entry_x_max.get())
        chromosome_proportion = float(entry_chromosome_length.get())
        maximize = var_maximize.get() == 1

        final_population, final_fitness = genetic_algorithm(population_size, generations, mutation_prob_individual, mutation_prob_gene, x_min, x_max, chromosome_proportion, maximize)

        best_index = np.argmax(final_fitness) if maximize else np.argmin(final_fitness)
        best_solution = binary_to_decimal(final_population[best_index], x_min, x_max)
        result_label.config(text=f'Mejor solución: x = {best_solution}, f(x) = {final_fitness[best_index]}')

        create_video_from_images('generation_plots', 'statistics_plots/generation_evolution.avi')
        
        # Guardar los resultados en un archivo CSV
        results_df = pd.DataFrame({
            'Individuo': final_population,
            'Valor del índice': range(len(final_population)),
            'Valor de x': [binary_to_decimal(ind, x_min, x_max) for ind in final_population],
            'Aptitud': final_fitness
        })
        results_df.to_csv('results.csv', index=False)

        # Mostrar la tabla en una ventana de la interfaz gráfica
        show_results_table(results_df)

    except ValueError as e:
        messagebox.showerror("Error de entrada", f"Por favor, ingrese valores válidos.\n\nDetalles del error: {e}")

def show_results_table(results_df):
    results_window = tk.Toplevel(root)
    results_window.title("Resultados")

    table = ttk.Treeview(results_window, columns=('Individuo', 'Valor del índice', 'Valor de x', 'Aptitud'), show='headings')
    table.heading('Individuo', text='Individuo')
    table.heading('Valor del índice', text='Valor del índice')
    table.heading('Valor de x', text='Valor de x')
    table.heading('Aptitud', text='Aptitud')

    for _, row in results_df.iterrows():
        table.insert('', 'end', values=(row['Individuo'], row['Valor del índice'], row['Valor de x'], row['Aptitud']))

    table.pack(expand=True, fill='both')

root = tk.Tk()
root.title("Algoritmo Genético")

# Parámetros de entrada
ttk.Label(root, text="Tamaño de la población:").grid(row=0, column=0, padx=10, pady=5)
entry_population_size = ttk.Entry(root)
entry_population_size.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Número de generaciones:").grid(row=1, column=0, padx=10, pady=5)
entry_generations = ttk.Entry(root)
entry_generations.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text="Probabilidad de mutación del individuo:").grid(row=2, column=0, padx=10, pady=5)
entry_mutation_prob_individual = ttk.Entry(root)
entry_mutation_prob_individual.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(root, text="Probabilidad de mutación del gen:").grid(row=3, column=0, padx=10, pady=5)
entry_mutation_prob_gene = ttk.Entry(root)
entry_mutation_prob_gene.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(root, text="Valor mínimo de x:").grid(row=4, column=0, padx=10, pady=5)
entry_x_min = ttk.Entry(root)
entry_x_min.grid(row=4, column=1, padx=10, pady=5)

ttk.Label(root, text="Valor máximo de x:").grid(row=5, column=0, padx=10, pady=5)
entry_x_max = ttk.Entry(root)
entry_x_max.grid(row=5, column=1, padx=10, pady=5)

ttk.Label(root, text="Longitud del cromosoma (proporción):").grid(row=6, column=0, padx=10, pady=5)
entry_chromosome_length = ttk.Entry(root)
entry_chromosome_length.grid(row=6, column=1, padx=10, pady=5)

ttk.Label(root, text="Maximización:").grid(row=7, column=0, padx=10, pady=5)
var_maximize = tk.IntVar()
ttk.Checkbutton(root, variable=var_maximize).grid(row=7, column=1, padx=10, pady=5)

print(entry_chromosome_length, entry_x_max, entry_x_min, entry_mutation_prob_gene, entry_mutation_prob_individual, entry_population_size, entry_generations)

ttk.Button(root, text="Ejecutar", command=run_algorithm).grid(row=8, column=0, columnspan=2, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(row=9, column=0, columnspan=2, pady=10)

root.mainloop()

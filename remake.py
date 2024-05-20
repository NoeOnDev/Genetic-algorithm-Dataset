import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd

# Crear carpetas si no existen
if not os.path.exists('generation_plots'):
    os.makedirs('generation_plots')
if not os.path.exists('statistics_plots'):
    os.makedirs('statistics_plots')

# Función a optimizar
def f(x):
    return x * np.cos(x)

# Convertir binario a decimal
def binary_to_decimal(binary_str, x_min, x_max):
    decimal = int(binary_str, 2)
    max_value = 2**len(binary_str) - 1
    return x_min + (decimal / max_value) * (x_max - x_min)

# Convertir decimal a binario
def decimal_to_binary(value, x_min, x_max, chromosome_length):
    max_value = 2**chromosome_length - 1
    normalized = int((value - x_min) / (x_max - x_min) * max_value)
    return f'{normalized:0{chromosome_length}b}'

# Inicialización de la población
def initialize_population(size, x_min, x_max, chromosome_length):
    population = []
    for _ in range(size):
        value = random.uniform(x_min, x_max)
        binary_str = decimal_to_binary(value, x_min, x_max, chromosome_length)
        population.append(binary_str)
    return population

# Evaluar la aptitud (fitness) de cada individuo
def evaluate_population(population, x_min, x_max):
    return np.array([f(binary_to_decimal(ind, x_min, x_max)) for ind in population])

# Seleccionar parejas de individuos (Estrategia A1)
def form_pairs(population):
    pairs = []
    n = len(population)
    for individual in population:
        m = random.randint(1, n)
        partners = random.sample(list(population), m)
        if individual in partners:
            partners.remove(individual)
        pairs.append((individual, partners))
    return pairs

# Cruza de información (Estrategia C2)
def crossover(parents):
    parent1, parent2 = parents
    num_points = random.randint(1, len(parent1) - 1)
    crossover_points = sorted(random.sample(range(1, len(parent1)), num_points))
    
    offspring = list(parent1)
    for i in range(len(crossover_points)):
        if i % 2 == 0:
            offspring[crossover_points[i]:] = parent2[crossover_points[i]:]
        else:
            offspring[:crossover_points[i]] = parent2[:crossover_points[i]]
    
    return ''.join(offspring)

# Mutar los individuos descendientes (Estrategia M2)
def mutate(individual, mutation_prob_individual, mutation_prob_gene):
    if random.random() < mutation_prob_individual:
        individual = list(individual)
        for i in range(len(individual)):
            if random.random() < mutation_prob_gene:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        individual = ''.join(individual)
    return individual

# Poda (Estrategia P2)
def prune_population(population, fitness, size, maximize):
    unique_population, unique_indices = np.unique(population, return_index=True)
    unique_fitness = fitness[unique_indices]
    
    if len(unique_population) > size:
        if maximize:
            sorted_indices = np.argsort(-unique_fitness)
        else:
            sorted_indices = np.argsort(unique_fitness)
        pruned_population = unique_population[sorted_indices][:size]
        pruned_fitness = unique_fitness[sorted_indices][:size]
    else:
        pruned_population = unique_population
        pruned_fitness = unique_fitness
    
    return pruned_population, pruned_fitness

# Bucle de optimización
def genetic_algorithm(population_size, generations, mutation_prob_individual, mutation_prob_gene, x_min, x_max, chromosome_proportion, maximize):
    max_chromosome_length = 20  # Longitud máxima del cromosoma (puede ajustarse según sea necesario)
    chromosome_length = int(chromosome_proportion * max_chromosome_length)
    
    population = initialize_population(population_size, x_min, x_max, chromosome_length)
    fitness = evaluate_population(population, x_min, x_max)
    
    # Listas para guardar los estadísticos
    best_fitnesses = []
    avg_fitnesses = []
    worst_fitnesses = []
    
    for generation in range(generations):
        new_population = []
        pairs = form_pairs(population)
        
        for individual, partners in pairs:
            for partner in partners:
                offspring = crossover([individual, partner])
                offspring = mutate(offspring, mutation_prob_individual, mutation_prob_gene)
                new_population.append(offspring)
        
        new_population = np.array(new_population)
        new_fitness = evaluate_population(new_population, x_min, x_max)
        
        combined_population = np.concatenate((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        
        # Guardar estadísticos de la población
        if maximize:
            best_individual = combined_population[np.argmax(combined_fitness)]
            best_fitness = np.max(combined_fitness)
        else:
            best_individual = combined_population[np.argmin(combined_fitness)]
            best_fitness = np.min(combined_fitness)
        
        avg_fitness = np.mean(combined_fitness)
        worst_fitness = np.min(combined_fitness) if maximize else np.max(combined_fitness)
        
        print(f'Generation {generation}: Best Individual = {binary_to_decimal(best_individual, x_min, x_max)} Fitness = {best_fitness}')
        
        # Guardar los estadísticos en las listas
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        worst_fitnesses.append(worst_fitness)
        
        # Crear y guardar la gráfica de la generación
        values = [binary_to_decimal(ind, x_min, x_max) for ind in combined_population]
        best_value = binary_to_decimal(best_individual, x_min, x_max)
        worst_value = binary_to_decimal(combined_population[np.argmin(combined_fitness)], x_min, x_max)
        
        frequencies, bin_edges = np.histogram(values, bins=20)
        plt.figure()
        plt.plot(bin_edges[:-1], frequencies, label='Histogram')
        plt.scatter([best_value], [0], color='red', label='Best', zorder=5)
        plt.scatter([worst_value], [0], color='blue', label='Worst', zorder=5)
        plt.scatter(values, np.zeros_like(values), color='green', label='Population', zorder=5, alpha=0.6)
        plt.title(f'Generation {generation}')
        plt.xlabel('x')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'generation_plots/generation_{generation}.png')
        plt.close()
        
        population, fitness = prune_population(combined_population, combined_fitness, population_size, maximize)
    
    # Crear y guardar la gráfica de los estadísticos
    plt.figure()
    plt.plot(best_fitnesses, label='Best')
    plt.plot(avg_fitnesses, label='Average')
    plt.plot(worst_fitnesses, label='Worst')
    plt.title('Fitness Statistics')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('statistics_plots/statistics.png')
    plt.close()
    
    return population, fitness

# Función para crear un video a partir de las imágenes generadas
def create_video_from_images(image_folder, output_video):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()

# Interfaz gráfica
def run_algorithm():
    try:
        population_size = int(entry_population_size.get())
        generations = int(entry_generations.get())
        mutation_prob_individual = float(entry_mutation_prob_individual.get())
        mutation_prob_gene = float(entry_mutation_prob_gene.get())
        x_min = float(entry_x_min.get())
        x_max = float(entry_x_max.get())
        chromosome_proportion = float(entry_chromosome_length.get())  # Interpreta como proporción
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

# Mostrar la tabla de resultados en una ventana de la interfaz gráfica
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

# Configuración de la interfaz gráfica
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

# Botón para ejecutar el algoritmo
ttk.Button(root, text="Ejecutar", command=run_algorithm).grid(row=8, column=0, columnspan=2, pady=10)

# Etiqueta para mostrar el resultado
result_label = ttk.Label(root, text="")
result_label.grid(row=9, column=0, columnspan=2, pady=10)

root.mainloop()

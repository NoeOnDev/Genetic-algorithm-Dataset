import numpy as np
import random
import matplotlib.pyplot as plt
import os

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
    return x_min + (decimal / (2**16 - 1)) * (x_max - x_min)

# Convertir decimal a binario
def decimal_to_binary(value, x_min, x_max):
    normalized = int((value - x_min) / (x_max - x_min) * (2**16 - 1))
    return f'{normalized:016b}'

# Inicialización de la población
def initialize_population(size, x_min, x_max):
    population = []
    for _ in range(size):
        value = random.uniform(x_min, x_max)
        binary_str = decimal_to_binary(value, x_min, x_max)
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
def prune_population(population, fitness, size):
    unique_population, unique_indices = np.unique(population, return_index=True)
    unique_fitness = fitness[unique_indices]
    
    if len(unique_population) > size:
        sorted_indices = np.argsort(-unique_fitness)
        pruned_population = unique_population[sorted_indices][:size]
        pruned_fitness = unique_fitness[sorted_indices][:size]
    else:
        pruned_population = unique_population
        pruned_fitness = unique_fitness
    
    return pruned_population, pruned_fitness

# Bucle de optimización
def genetic_algorithm(population_size, generations, mutation_prob_individual, mutation_prob_gene, x_min, x_max):
    population = initialize_population(population_size, x_min, x_max)
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
        best_individual = combined_population[np.argmax(combined_fitness)]
        best_fitness = np.max(combined_fitness)
        avg_fitness = np.mean(combined_fitness)
        worst_fitness = np.min(combined_fitness)
        
        print(f'Generation {generation}: Best Individual = {binary_to_decimal(best_individual, x_min, x_max)} Fitness = {best_fitness}')
        
        # Guardar los estadísticos en las listas
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        worst_fitnesses.append(worst_fitness)
        
        # Crear y guardar la gráfica de la generación
        values = [binary_to_decimal(ind, x_min, x_max) for ind in combined_population]
        frequencies, bin_edges = np.histogram(values, bins=20)
        plt.figure()
        plt.plot(bin_edges[:-1], frequencies)
        plt.title(f'Generation {generation}')
        plt.xlabel('x')
        plt.ylabel('Frequency')
        plt.savefig(f'generation_plots/generation_{generation}.png')
        plt.close()
        
        population, fitness = prune_population(combined_population, combined_fitness, population_size)
    
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

population_size = int(input("Tamaño de la población: "))
generations = int(input("Número de generaciones: "))
mutation_prob_individual = float(input("Probabilidad de mutación del individuo: "))
mutation_prob_gene = float(input("Probabilidad de mutación del gen: "))
x_min = float(input("Valor mínimo de x: "))
x_max = float(input("Valor máximo de x: "))

final_population, final_fitness = genetic_algorithm(population_size, generations, mutation_prob_individual, mutation_prob_gene, x_min, x_max)

best_index = np.argmax(final_fitness)
best_solution = binary_to_decimal(final_population[best_index], x_min, x_max)
print(f'Mejor solución encontrada: x = {best_solution}, f(x) = {final_fitness[best_index]}')

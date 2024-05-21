import numpy as np
import random
import argparse

# Función para convertir binario a decimal
def binary_to_decimal(binary_str, x_min, x_max):
    integer_value = int(binary_str, 2)
    return x_min + (x_max - x_min) * integer_value / (2**len(binary_str) - 1)

# Función para convertir decimal a binario
def decimal_to_binary(value, x_min, x_max, num_bits=16):
    integer_value = int((value - x_min) / (x_max - x_min) * (2**num_bits - 1))
    return format(integer_value, f'0{num_bits}b')

# Inicialización de la población
def initialize_population(pop_size, x_min, x_max, num_bits=16):
    population = []
    for _ in range(pop_size):
        value = random.uniform(x_min, x_max)
        binary_str = decimal_to_binary(value, x_min, x_max, num_bits)
        population.append(binary_str)
    return population

# Función de aptitud
def fitness_function(binary_str, x_min, x_max, maximize=True):
    x = binary_to_decimal(binary_str, x_min, x_max)
    fitness = x * np.cos(x)
    return fitness if maximize else -fitness

# Formación de parejas
def form_pairs(population):
    pairs = []
    pop_size = len(population)
    for i in range(pop_size):
        m = random.randint(0, pop_size - 1)
        mates = random.sample(range(pop_size), m)
        if i in mates:
            mates.remove(i)
        pairs.append((i, mates))
    return pairs

# Cruza de información
def crossover(parent1, parent2, num_bits=16):
    num_points = random.randint(1, num_bits - 1)
    crossover_points = sorted(random.sample(range(1, num_bits), num_points))
    child1, child2 = list(parent1), list(parent2)
    for i in range(0, len(crossover_points), 2):
        if i + 1 < len(crossover_points):
            child1[crossover_points[i]:crossover_points[i+1]] = parent2[crossover_points[i]:crossover_points[i+1]]
            child2[crossover_points[i]:crossover_points[i+1]] = parent1[crossover_points[i]:crossover_points[i+1]]
    return ''.join(child1), ''.join(child2)

# Crear descendencia
def create_offspring(population, pairs, num_bits=16):
    offspring = []
    for i, mates in pairs:
        parent1 = population[i]
        for mate in mates:
            parent2 = population[mate]
            child1, child2 = crossover(parent1, parent2, num_bits)
            offspring.append(child1)
            offspring.append(child2)
    return offspring

# Mutación
def mutate(individual, mutation_rate_individual, mutation_rate_gene, num_bits=16):
    if random.random() < mutation_rate_individual:
        individual = list(individual)
        for i in range(num_bits):
            if random.random() < mutation_rate_gene:
                individual[i] = '1' if individual[i] == '0' else '0'
        individual = ''.join(individual)
    return individual

def apply_mutations(offspring, mutation_rate_individual, mutation_rate_gene, num_bits=16):
    return [mutate(individual, mutation_rate_individual, mutation_rate_gene, num_bits) for individual in offspring]

# Podar población
def prune_population(population, fitness, max_size, best_individual):
    unique_population, unique_indices = np.unique(population, return_index=True)
    unique_fitness = [fitness[i] for i in unique_indices]
    
    best_index = np.argmax(unique_fitness)
    best_individual = unique_population[best_index]
    
    indices = list(range(len(unique_population)))
    indices.remove(best_index)
    random.shuffle(indices)
    indices = indices[:max_size-1]
    
    new_population = [best_individual] + [unique_population[i] for i in indices]
    new_fitness = [unique_fitness[best_index]] + [unique_fitness[i] for i in indices]
    
    return new_population, new_fitness

# Argumentos del programa
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Algoritmo Genético para maximizar y minimizar funciones 2D.')
    parser.add_argument('--pop_size', type=int, default=20, help='Tamaño de la población inicial')
    parser.add_argument('--max_pop_size', type=int, default=30, help='Tamaño máximo de la población')
    parser.add_argument('--x_min', type=float, default=-10, help='Límite inferior de x')
    parser.add_argument('--x_max', type=float, default=10, help='Límite superior de x')
    parser.add_argument('--mutation_rate_individual', type=float, default=0.1, help='Probabilidad de mutación del individuo')
    parser.add_argument('--mutation_rate_gene', type=float, default=0.05, help='Probabilidad de mutación del gen')
    parser.add_argument('--generations', type=int, default=50, help='Número de generaciones')
    parser.add_argument('--maximize', action='store_true', help='Maximizar la función en lugar de minimizarla')

    args = parser.parse_args()

    # Inicialización
    population = initialize_population(args.pop_size, args.x_min, args.x_max)
    fitness = [fitness_function(ind, args.x_min, args.x_max, args.maximize) for ind in population]

    for generation in range(args.generations):
        pairs = form_pairs(population)
        offspring = create_offspring(population, pairs)
        mutated_offspring = apply_mutations(offspring, args.mutation_rate_individual, args.mutation_rate_gene)
        offspring_fitness = [fitness_function(ind, args.x_min, args.x_max, args.maximize) for ind in mutated_offspring]

        combined_population = population + mutated_offspring
        combined_fitness = fitness + offspring_fitness

        unique_population, unique_indices = np.unique(combined_population, return_index=True)
        unique_fitness = [combined_fitness[i] for i in unique_indices]

        best_index = np.argmax(unique_fitness) if args.maximize else np.argmin(unique_fitness)
        best_individual = unique_population[best_index]

        population, fitness = prune_population(unique_population, unique_fitness, args.pop_size, best_individual)

        print(f"Generación {generation + 1}: Mejor individuo = {binary_to_decimal(best_individual, args.x_min, args.x_max)}, Fitness = {unique_fitness[best_index]}")

    print("Optimización finalizada.")

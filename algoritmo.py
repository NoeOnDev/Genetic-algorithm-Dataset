import numpy as np

POP_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
BOUNDS = [(-10, 10), (-10, 10)]
TOURNAMENT_SIZE = 3

def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        x = np.random.uniform(BOUNDS[0][0], BOUNDS[0][1])
        y = np.random.uniform(BOUNDS[1][0], BOUNDS[1][1])
        population.append((x, y))
    return np.array(population)

def fitness_function(individual):
    x, y = individual
    return f(x, y)

def tournament_selection(population, fitness):
    selected = []
    for _ in range(len(population)):
        aspirants = np.random.choice(len(population), TOURNAMENT_SIZE)
        best = aspirants[np.argmax(fitness[aspirants])]
        selected.append(population[best])
    return np.array(selected)

def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = np.random.uniform(BOUNDS[i][0], BOUNDS[i][1])
    return individual

def genetic_algorithm(f):
    population = initialize_population()
    best_solution = None
    best_fitness = -np.inf

    for gen in range(GENERATIONS):
        fitness = np.array([fitness_function(ind) for ind in population])
        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fitness:
            best_fitness = fitness[best_index]
            best_solution = population[best_index]
        
        selected_population = tournament_selection(population, fitness)
        next_population = []

        for i in range(0, POP_SIZE, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))
        
        population = np.array(next_population)
        
        print(f"Generation {gen+1}, Best Fitness: {best_fitness}")

    return best_solution, best_fitness

def f(x, y):
    return -((x-2)**2 + (y-3)**2) + 10

best_solution, best_fitness = genetic_algorithm(f)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
import numpy as np
import random
import matplotlib.pyplot as plt

class AlgoritmoGeneticoDTO:
    def __init__(self, tam_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen, x_min, x_max, num_generaciones, modo):
        self.tam_poblacion = tam_poblacion
        self.max_poblacion = max_poblacion
        self.prob_mutacion_individuo = prob_mutacion_individuo
        self.prob_mutacion_gen = prob_mutacion_gen
        self.x_min = x_min
        self.x_max = x_max
        self.num_generaciones = num_generaciones
        self.modo = modo

def funcion_fitness(x):
    return x * np.cos(x)

def inicializar_poblacion(tam_poblacion, x_min, x_max):
    return np.random.uniform(x_min, x_max, tam_poblacion)

def evaluar_poblacion(poblacion, funcion_fitness):
    return np.array([funcion_fitness(ind) for ind in poblacion])

def seleccionar_parejas(poblacion, n):
    num_individuos = len(poblacion)
    padres = []
    for i in range(num_individuos):
        m = random.randint(0, min(n, num_individuos - 1))
        indices_seleccionados = random.sample(range(num_individuos), m)
        if i in indices_seleccionados:
            indices_seleccionados.remove(i)
        padres.append((i, indices_seleccionados))
    return padres

def cruce(padres, poblacion):
    descendientes = []
    for padre, parejas in padres:
        for pareja in parejas:
            if pareja != padre:
                padre1 = poblacion[padre]
                padre2 = poblacion[pareja]
                num_puntos = random.randint(1, 15)
                puntos = sorted(random.sample(range(1, 16), num_puntos))
                hijo = cruce_bits(padre1, padre2, puntos)
                descendientes.append(hijo)
    return np.array(descendientes)

def cruce_bits(padre1, padre2, puntos):
    bin_padre1 = list(bin(int(padre1) & 0xFFFF)[2:].zfill(16))
    bin_padre2 = list(bin(int(padre2) & 0xFFFF)[2:].zfill(16))
    hijo = bin_padre1.copy()
    for i, punto in enumerate(puntos):
        if i % 2 == 0:
            hijo[punto:] = bin_padre2[punto:]
        else:
            hijo[punto:] = bin_padre1[punto:]
    return int(''.join(hijo), 2)

def mutar(descendientes, prob_mutacion_individuo, prob_mutacion_gen):
    for idx in range(len(descendientes)):
        if np.random.rand() < prob_mutacion_individuo:
            bin_descendiente = list(bin(int(descendientes[idx]) & 0xFFFF)[2:].zfill(16))
            for gen in range(len(bin_descendiente)):
                if np.random.rand() < prob_mutacion_gen:
                    swap_idx = random.randint(0, len(bin_descendiente) - 1)
                    bin_descendiente[gen], bin_descendiente[swap_idx] = bin_descendiente[swap_idx], bin_descendiente[gen]
            descendientes[idx] = int(''.join(bin_descendiente), 2)
    return descendientes

def podar_poblacion(poblacion, descendientes, funcion_fitness, modo):
    nueva_poblacion = np.concatenate((poblacion, descendientes))
    nueva_poblacion = np.unique(nueva_poblacion)
    fitness_nueva_poblacion = evaluar_poblacion(nueva_poblacion, funcion_fitness)
    
    if modo == 'maximizar':
        indices_ordenados = np.argsort(fitness_nueva_poblacion)[::-1]
    else:
        indices_ordenados = np.argsort(fitness_nueva_poblacion)
    
    mejor_individuo = nueva_poblacion[indices_ordenados[0]]
    indices_ordenados = indices_ordenados[:len(poblacion) - 1]
    nueva_poblacion = nueva_poblacion[indices_ordenados]
    nueva_poblacion = np.append(nueva_poblacion, mejor_individuo)
    return nueva_poblacion

def algoritmo_genetico(dto):
    poblacion = inicializar_poblacion(dto.tam_poblacion, dto.x_min, dto.x_max)
    mejores_por_generacion = []
    peores_por_generacion = []
    promedio_por_generacion = []
    
    for generacion in range(dto.num_generaciones):
        fitness = evaluar_poblacion(poblacion, funcion_fitness)
        
        mejor_individuo_idx = np.argmax(fitness) if dto.modo == 'maximizar' else np.argmin(fitness)
        peor_individuo_idx = np.argmin(fitness) if dto.modo == 'maximizar' else np.argmax(fitness)
        mejor_individuo = poblacion[mejor_individuo_idx]
        peor_individuo = poblacion[peor_individuo_idx]
        promedio_fitness = np.mean(fitness)
        
        mejores_por_generacion.append((mejor_individuo, fitness[mejor_individuo_idx]))
        peores_por_generacion.append((peor_individuo, fitness[peor_individuo_idx]))
        promedio_por_generacion.append(promedio_fitness)
        
        print(f"Generación {generacion}: Mejor individuo: {mejor_individuo:.2f}, Aptitud: {fitness[mejor_individuo_idx]:.2f}")
        
        padres = seleccionar_parejas(poblacion, dto.max_poblacion)
        descendientes_cruza = cruce(padres, poblacion)
        descendientes_mutados = mutar(descendientes_cruza, dto.prob_mutacion_individuo, dto.prob_mutacion_gen)
        poblacion = podar_poblacion(poblacion, descendientes_mutados, funcion_fitness, dto.modo)
    
    graficar_resultados(mejores_por_generacion, peores_por_generacion, promedio_por_generacion)
    
    fitness_final = evaluar_poblacion(poblacion, funcion_fitness)
    mejor_solucion_idx = np.argmax(fitness_final) if dto.modo == 'maximizar' else np.argmin(fitness_final)
    return poblacion[mejor_solucion_idx]

def graficar_resultados(mejores_por_generacion, peores_por_generacion, promedio_por_generacion):
    generaciones = range(len(mejores_por_generacion))
    mejores_aptitudes = [ind[1] for ind in mejores_por_generacion]
    peores_aptitudes = [ind[1] for ind in peores_por_generacion]
    
    plt.figure(figsize=(10, 5))
    plt.plot(generaciones, mejores_aptitudes, label='Mejor Aptitud')
    plt.plot(generaciones, peores_aptitudes, label='Peor Aptitud')
    plt.plot(generaciones, promedio_por_generacion, label='Aptitud Promedio')
    
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.title('Evolución de la Aptitud')
    plt.legend()
    plt.grid(True)
    plt.show()

dto = AlgoritmoGeneticoDTO(
    tam_poblacion=10,
    max_poblacion=20,
    prob_mutacion_individuo=0.1,
    prob_mutacion_gen=0.05,
    x_min=-10,
    x_max=10,
    num_generaciones=50,
    modo='minimizar'  # O 'minimizar'
)

mejor_solucion = algoritmo_genetico(dto)
print(f"Mejor solución encontrada: {mejor_solucion:.2f}, Aptitud: {funcion_fitness(mejor_solucion):.2f}")

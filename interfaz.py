import subprocess

# Definir los parámetros de prueba
args = {
    'tam_poblacion': 10,
    'tam_max_poblacion': 50,
    'x_min': -10,
    'x_max': 40,
    'tasa_mutacion_individuo': 0.7,
    'tasa_mutacion_gen': 0.6,
    'generaciones': 100,
    'maximizar': 'store_false',  # Cambiar a False para minimizar
    'directorio_graficas': 'graficas_minimizar',
    'directorio_evolucion': 'evolucion_minimizar',
    'output_video': 'evolucion_minimizar.mp4'
}

cmd = [
    'python', 'algoritmo_genetico.py',
    '--tam_poblacion', str(args['tam_poblacion']),
    '--tam_max_poblacion', str(args['tam_max_poblacion']),
    '--x_min', str(args['x_min']),
    '--x_max', str(args['x_max']),
    '--tasa_mutacion_individuo', str(args['tasa_mutacion_individuo']),
    '--tasa_mutacion_gen', str(args['tasa_mutacion_gen']),
    '--generaciones', str(args['generaciones']),
    '--directorio_graficas', args['directorio_graficas'],
    '--directorio_evolucion', args['directorio_evolucion'],
    '--output_video', args['output_video']
]

if not args['maximizar']:
    cmd.append('--maximizar')

subprocess.run(cmd)
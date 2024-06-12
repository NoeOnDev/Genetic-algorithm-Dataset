import os
import re
import cv2
import matplotlib.pyplot as plt

def crear_video():
    img_dir = "imagenes_graficas_generadas"
    video_dir = "video_generado"
    os.makedirs(video_dir, exist_ok=True)
    video_filename = os.path.join(video_dir, 'generations_video.mp4')

    images = [img for img in os.listdir(img_dir) if img.endswith(".png")]
    images.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    frame = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 3, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_dir, image)))

    cv2.destroyAllWindows()
    video.release()

def crear_grafica_error(norm_errores, promedio_errores, peores):
    img_dir = "imagen_errores"
    os.makedirs(img_dir, exist_ok=True)
    
    def save_plots(norm_errores, promedio_errores, peores):
        plt.figure(figsize=(12, 8))
        plt.plot(norm_errores, color='blue', label='Mejores de cada generacion')
        plt.plot(promedio_errores, color='black', label='Promedio de cada generacion')
        plt.plot(peores, color='red', label='Peores de cada generacion')
        plt.title('Evolución de las aptitudes de la población')
        plt.xlabel('Generación')
        plt.ylabel('Aptitud de la población')
        plt.grid(True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
        plt.tight_layout()
        filename = os.path.join(img_dir, 'errores.png')
        plt.savefig(filename)
        plt.show()
    
    save_plots(norm_errores, promedio_errores, peores)

def crear_graficas_constante(a, b, c, d, e, f):
    img_dir = "imagen_constantes"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, c, d, e, f):
        x = range(len(a))
        plt.figure(figsize=(12, 8))
        plt.scatter(x, a, color='blue', label='')
        plt.plot(x, a, color='blue', label='A')
        plt.scatter(x, b, color='green', label='')
        plt.plot(x, b, color='green', label='B')
        plt.scatter(x, c, color='red', label='')
        plt.plot(x, c, color='red', label='C')
        plt.scatter(x, d, color='gray', label='')
        plt.plot(x, d, color='gray', label='D')
        plt.scatter(x, e, color='black', label='')
        plt.plot(x, e, color='black', label='E')
        plt.scatter(x, f, color='blue', label='')
        plt.plot(x, f, color='blue', label='F')
        plt.title('Evolución de los parámetros')
        plt.xlabel('Generación')
        plt.ylabel('Parámetros del mejor individuo')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{img_dir}/constantes.png"
        plt.savefig(filename)
        plt.show()

    save_plots(a, b, c, d, e, f)

def generar_nombre_archivo_generacion(num_generacion):
    return f"generation_{num_generacion:03d}.png"

def crear_grafica(yd, fx, i):
    img_dir = "imagenes_graficas_generadas"
    os.makedirs(img_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.plot(fx, color='green', label='Resultado obtenido')
    plt.plot(yd, color='black', label='Resultado deseado')
    plt.scatter(range(len(fx)), fx, color='green', s=0, label='Resultados obtenidos')
    plt.scatter(range(len(yd)), yd, color='black', s=0, label='Resultados deseados')
    plt.title(f'Generación {i}')
    plt.xlabel('Cantidad de generaciones')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.tight_layout()
    filename = os.path.join(img_dir, generar_nombre_archivo_generacion(i))
    plt.savefig(filename)
    plt.close()

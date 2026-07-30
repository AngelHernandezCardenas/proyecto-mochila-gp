import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import random
import os

# GRÁFICA 1: BANDA DE DESVIACIÓN EVOLUTIVA

def graficar_convergencia_con_sombra(archivo_csv, titulo, color_linea, color_sombra):
    try:
        df = pd.read_csv(archivo_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_csv}.")
        return

    generaciones = df['gen']
    promedio = df['Promedio']
    desviacion = df['Desviacion']
    maxima = df['Max_Ganancia']

    plt.figure(figsize=(10, 6))
    
    plt.fill_between(generaciones, promedio - desviacion, promedio + desviacion, 
                     color=color_sombra, alpha=0.3, label='Desviación Estándar (Caos Genético)')
    
    plt.plot(generaciones, promedio, color=color_linea, linewidth=2.5, marker='o', 
             label='Aptitud Promedio de la Población')
    
    plt.plot(generaciones, maxima, color='black', linestyle='--', linewidth=1.5, 
             label='Límite de Ganancia Máxima (Elitismo)')

    plt.title(titulo, fontsize=14, pad=15)
    plt.xlabel('Generaciones Evolutivas', fontsize=12)
    plt.ylabel('Ganancia Matemática (Fitness)', fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    # GUARDAR COMO IMAGEN EN LUGAR DE MOSTRAR VENTANA
    nombre_img = "Grafica_Convergencia.png"
    plt.savefig(nombre_img, dpi=300)
    print(f"-> ¡Éxito! Gráfica guardada como: {nombre_img}")
    plt.close()


# GRÁFICA 2: MAPA DE AGRUPAMIENTO K-MEANS

def graficar_mapa_kmeans(num_instancias=30, num_objetos=50):
    pesos_promedio = []
    ganancias_promedio = []
    
    for _ in range(num_instancias):
        w_list = [random.uniform(1.0, 20.0) for _ in range(num_objetos)]
        p_list = [random.uniform(10.0, 100.0) for _ in range(num_objetos)]
        pesos_promedio.append(np.mean(w_list))
        ganancias_promedio.append(np.mean(p_list))
        
    caracteristicas = list(zip(pesos_promedio, ganancias_promedio))
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(caracteristicas)
    centros = kmeans.cluster_centers_
    
    plt.figure(figsize=(10, 6))
    
    c0_w = [caracteristicas[i][0] for i in range(len(caracteristicas)) if etiquetas[i] == 0]
    c0_p = [caracteristicas[i][1] for i in range(len(caracteristicas)) if etiquetas[i] == 0]
    
    c1_w = [caracteristicas[i][0] for i in range(len(caracteristicas)) if etiquetas[i] == 1]
    c1_p = [caracteristicas[i][1] for i in range(len(caracteristicas)) if etiquetas[i] == 1]
    
    plt.scatter(c0_w, c0_p, color='#1f77b4', s=100, edgecolors='w', label='Clúster 0 (Mochilas Tipo A)')
    plt.scatter(c1_w, c1_p, color='#ff7f0e', s=100, edgecolors='w', label='Clúster 1 (Mochilas Tipo B)')
    
    plt.scatter(centros[:, 0], centros[:, 1], color='red', marker='X', s=200, label='Centroides (Núcleos K-Means)')
    
    plt.title('Clasificación Inteligente de Instancias Vectoriales (K-Means)', fontsize=14, pad=15)
    plt.xlabel('Peso Promedio de los Objetos (W)', fontsize=12)
    plt.ylabel('Ganancia Promedio de los Objetos (P)', fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # GUARDAR COMO IMAGEN EN LUGAR DE MOSTRAR VENTANA
    nombre_img = "Grafica_Mapa_KMeans.png"
    plt.savefig(nombre_img, dpi=300)
    print(f"-> ¡Éxito! Gráfica guardada como: {nombre_img}")
    plt.close()

if __name__ == "__main__":
    print("Iniciando generación de imágenes...")
    
    # Asegúrate de que el CSV de la Fase 5 esté en la misma carpeta
    graficar_convergencia_con_sombra('Fase5_bitacora_cluster_0.csv', 
                                     'Convergencia Evolutiva y Reducción del Caos (Clúster 0)', 
                                     '#2ca02c', '#98df8a')
    
    graficar_mapa_kmeans()
    
    print("\nPROCESO TERMINADO. Revisa tu carpeta para ver las nuevas imágenes PNG.")
import operator                                  # Operadores matematicos basicos
import random                                    # Generacion de numeros aleatorios
import numpy as np                               # Manejo de arreglos y matematicas
from sklearn.cluster import KMeans               # Libreria para el amontonamiento inteligente
from typing import List, Tuple
# Importar clases necesarias (Asumimos que Item, KnapsackState, KnapsackInstance están accesibles)

# 1. Definicion de funciones matematicas seguras
def div_segura(izq, der):                        # Evita divisiones por cero en los arboles
    if der == 0:                                 # Si el denominador (peso) es cero
        return 1.0                               # Retorna un valor neutral para no quebrar el programa
    return izq / der                             # Retorna la division normal

# 2. Funcion de Clasificacion (Separación de instancias)
def clasificar_instancias(lista_instancias: List['KnapsackInstance'], num_clusters: int = 2) -> dict:
    caracteristicas = []                         # Lista para extraer rasgos matematicos de los problemas
    for instancia in lista_instancias:            # Recorre las bases de datos de prueba
        promedio_p = np.mean([e.profit for e in instancia.items]) # Extrae la ganancia promedio de la instancia
        promedio_w = np.mean([e.weight for e in instancia.items]) # Extrae el peso promedio de la instancia
        caracteristicas.append([promedio_p, promedio_w]) # Guarda los rasgos en un vector
        
    # Ejecucion de K-Means para clasificar las mochilas por similitud matematica
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) 
    etiquetas = kmeans.fit_predict(caracteristicas) # Ejecuta K-Means y etiqueta cada problema
    
    print("Agrupamiento K-Means completado. Etiquetas:", etiquetas) 
    
    # Creacion de diccionarios para separar los problemas segun su etiqueta de cluster
    clusters = {i: [] for i in range(num_clusters)}
    for idx, etiqueta in enumerate(etiquetas):
        clusters[etiqueta].append(lista_instancias[idx])
        
    return clusters # Retorna la estructura de clústeres

# Bloque principal de prueba (para verificar que el módulo funciona)
if __name__ == "__main__":
    # Crear instancias dummy para probar solo el clustering
    class DummyItem:
        def __init__(self, p, w): self.profit = p; self.weight = w
    class DummyInstance:
        def __init__(self, id, items): self.instance_id = id; self.items = items
    
    dummy_items1 = [DummyItem(10, 5), DummyItem(40, 4)]
    dummy_items2 = [DummyItem(15, 2), DummyItem(30, 8)]
    
    base_de_datos_prueba = [
        DummyInstance("Inst_A", dummy_items1),
        DummyInstance("Inst_B", dummy_items2)
    ]
    
    clusters_generados = clasificar_instancias(base_de_datos_prueba, num_clusters=2)
    print("\nClusteres resultantes:", clusters_generados)
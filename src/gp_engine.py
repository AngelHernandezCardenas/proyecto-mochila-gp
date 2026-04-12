import sys
sys.path.insert(0, 'src')
import operator
import random
import multiprocessing
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
from sklearn.cluster import KMeans
from knapsack import Item, KnapsackState, KnapsackInstance

# Identificador de la iteracion final
FASE_ACTUAL = 5

# 1. Definicion de funciones matematicas seguras para el arbol
def div_segura(izq, der):
    # Prevencion de division por cero asignando un valor neutral
    if der == 0:
        return 1.0
    return izq / der

# 2. Configuracion de la Programacion Genetica (GP)
# Definicion del conjunto de terminales y operaciones logicas con 3 entradas
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura, 2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')

# Creacion de las estructuras para maximizar la aptitud (Fitness)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Registro de herramientas fundamentales de DEAP
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 3. Funcion de evaluacion de la hiper-heuristica (Simulacion de empaquetado)
def evaluar_hiper_heuristica(individuo, instancia):
    # Traduccion del arbol matematico a una funcion ejecutable
    rutina_puntuacion = toolbox.compile(expr=individuo)
    mochila = KnapsackState(capacity=instancia.capacity)
    
    items_puntuados = []
    # Evaluacion de cada objeto disponible usando la formula inventada por la IA
    for item in instancia.items:
        puntuacion = rutina_puntuacion(item.profit, item.weight, item.ratio)
        items_puntuados.append((puntuacion, item))
        
    # Ordenamiento de los objetos priorizando las mejores calificaciones
    items_puntuados.sort(key=lambda x: x[0], reverse=True)
    
    # Empaquetado fisico respetando la capacidad de la mochila
    for puntuacion, item in items_puntuados:
        mochila.pack(item)
        
    # Penalizacion por longitud (Bloat) para ahorrar consumo de memoria RAM
    penalizacion_longitud = len(individuo) * 0.01
    
    # Se retorna la ganancia final ajustada
    return mochila.current_profit - penalizacion_longitud,

# 4. Operadores Geneticos y control de crecimiento
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

# 5. Generador Estocastico de bases de datos
def generar_base_datos_aleatoria(num_instancias=10, num_objetos=50):
    instancias = []
    # Creacion de multiples problemas con capacidades y objetos totalmente aleatorios
    for i in range(num_instancias):
        capacidad = random.uniform(50.0, 150.0)
        objetos = []
        for j in range(num_objetos):
            peso = random.uniform(1.0, 20.0)
            ganancia = random.uniform(10.0, 100.0)
            objetos.append(Item(j, peso, ganancia))
        instancias.append(KnapsackInstance(f"Inst_{i}", capacidad, objetos))
    return instancias

# 6. Agrupamiento y Bucle Evolutivo Paralelo
def clasificar_y_evolucionar(lista_instancias, num_clusters=2, generaciones=20):
    caracteristicas = []
    
    # Extraccion de promedios de peso y ganancia para alimentar al modelo K-Means
    for instancia in lista_instancias:
        promedio_p = np.mean([item.profit for item in instancia.items])
        promedio_w = np.mean([item.weight for item in instancia.items])
        caracteristicas.append([promedio_p, promedio_w])
        
    # Ejecucion de K-Means para clasificar las mochilas por similitud matematica
    kmeans = KMeans(n_clusters=num_clusters, random_state=None, n_init=10)
    etiquetas = kmeans.fit_predict(caracteristicas)
    
    # Creacion de diccionarios para separar los problemas segun su etiqueta de cluster
    clusters = {i: [] for i in range(num_clusters)}
    for idx, etiqueta in enumerate(etiquetas):
        clusters[etiqueta].append(lista_instancias[idx])
        
    mejores_reglas = {}
    
    # Activacion del multiprocessing para acelerar el algoritmo usando todos los nucleos del CPU
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    for cluster_id, instancias_cluster in clusters.items():
        if not instancias_cluster:
            continue
            
        print(f"\nIniciando evolucion paralela para el Cluster {cluster_id}")
        poblacion = toolbox.population(n=50)
        instancia_prueba = instancias_cluster[0]
        
        # Registro de la instancia actual en la funcion de evaluacion
        toolbox.register("evaluate", evaluar_hiper_heuristica, instancia=instancia_prueba)
        
        # Configuracion de las estadisticas a recolectar durante las generaciones
        estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0])
        estadisticas.register("Promedio", np.mean)
        estadisticas.register("Max_Ganancia", np.max)
        estadisticas.register("Desviacion", np.std)
        
        # Herramienta para guardar al mejor individuo historico (Elitismo)
        salon_fama = tools.HallOfFame(1)
        
        # Ejecucion del algoritmo generacional de DEAP
        poblacion_final, bitacora = algorithms.eaSimple(
            poblacion, toolbox, cxpb=0.7, mutpb=0.2, ngen=generaciones, 
            stats=estadisticas, halloffame=salon_fama, verbose=True
        )
        
        mejor_individuo = salon_fama[0]
        mejores_reglas[cluster_id] = str(mejor_individuo)
        
        # Automatizacion de exportacion a CSV usando Pandas
        df_log = pd.DataFrame(bitacora)
        df_log.to_csv(f"Fase{FASE_ACTUAL}_bitacora_cluster_{cluster_id}.csv", index=False)
        
    # Cierre y limpieza de los hilos de procesamiento
    pool.close()
    pool.join()
    
    # Guardado automatico de las formulas ganadoras en archivo de texto plano
    with open(f"Fase{FASE_ACTUAL}_mejores_reglas.txt", "w") as archivo_texto:
        archivo_texto.write(f"REPORTE DE FORMULAS EVOLUTIVAS - FASE {FASE_ACTUAL}\n")
        for cluster_id, regla in mejores_reglas.items():
            archivo_texto.write(f"Cluster {cluster_id}:\n{regla}\n\n")
            
    return mejores_reglas

# Bloque de ejecucion principal
if __name__ == "__main__":
    # Inyeccion de estres computacional: 10 problemas con 50 objetos aleatorios
    base_de_datos = generar_base_datos_aleatoria(num_instancias=10, num_objetos=50)
    clasificar_y_evolucionar(base_de_datos, num_clusters=2, generaciones=20)
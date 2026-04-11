import operator
import random
import multiprocessing
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
from sklearn.cluster import KMeans
from knapsack import Item, KnapsackState, KnapsackInstance

#FASE_ACTUAL = 4

def div_segura(izq, der):
    if der == 0:
        return 1.0
    return izq / der

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura, 2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluar_hiper_heuristica(individuo, instancia):
    rutina_puntuacion = toolbox.compile(expr=individuo)
    mochila = KnapsackState(capacity=instancia.capacity)
    
    items_puntuados = []
    for item in instancia.items:
        puntuacion = rutina_puntuacion(item.profit, item.weight, item.ratio)
        items_puntuados.append((puntuacion, item))
        
    items_puntuados.sort(key=lambda x: x[0], reverse=True)
    
    for puntuacion, item in items_puntuados:
        mochila.pack(item)
        
    penalizacion_longitud = len(individuo) * 0.01
    return mochila.current_profit - penalizacion_longitud,

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

def generar_base_datos_aleatoria(num_instancias=10, num_objetos=50):
    instancias = []
    for i in range(num_instancias):
        capacidad = random.uniform(50.0, 150.0)
        objetos = []
        for j in range(num_objetos):
            peso = random.uniform(1.0, 20.0)
            ganancia = random.uniform(10.0, 100.0)
            objetos.append(Item(j, peso, ganancia))
        instancias.append(KnapsackInstance(f"Inst_{i}", capacidad, objetos))
    return instancias

def clasificar_y_evolucionar(lista_instancias, num_clusters=2, generaciones=20):
    caracteristicas = []
    for instancia in lista_instancias:
        promedio_p = np.mean([item.profit for item in instancia.items])
        promedio_w = np.mean([item.weight for item in instancia.items])
        caracteristicas.append([promedio_p, promedio_w])
        
    kmeans = KMeans(n_clusters=num_clusters, random_state=None, n_init=10)
    etiquetas = kmeans.fit_predict(caracteristicas)
    
    clusters = {i: [] for i in range(num_clusters)}
    for idx, etiqueta in enumerate(etiquetas):
        clusters[etiqueta].append(lista_instancias[idx])
        
    mejores_reglas = {}
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    for cluster_id, instancias_cluster in clusters.items():
        if not instancias_cluster:
            continue
            
        print(f"\nIniciando evolución paralela para el Clúster {cluster_id}")
        poblacion = toolbox.population(n=50)
        instancia_prueba = instancias_cluster[0]
        
        toolbox.register("evaluate", evaluar_hiper_heuristica, instancia=instancia_prueba)
        
        estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0])
        estadisticas.register("Promedio", np.mean)
        estadisticas.register("Max_Ganancia", np.max)
        estadisticas.register("Desviacion", np.std)
        
        salon_fama = tools.HallOfFame(1)
        
        poblacion_final, bitacora = algorithms.eaSimple(
            poblacion, toolbox, cxpb=0.7, mutpb=0.2, ngen=generaciones, 
            stats=estadisticas, halloffame=salon_fama, verbose=True
        )
        
        mejor_individuo = salon_fama[0]
        mejores_reglas[cluster_id] = str(mejor_individuo)
        
        df_log = pd.DataFrame(bitacora)
        df_log.to_csv(f"Fase{FASE_ACTUAL}_bitacora_cluster_{cluster_id}.csv", index=False)
        
    pool.close()
    pool.join()
    
    with open(f"Fase{FASE_ACTUAL}_mejores_reglas.txt", "w") as archivo_texto:
        archivo_texto.write(f"REPORTE DE FÓRMULAS EVOLUTIVAS - FASE {FASE_ACTUAL}\n")
        for cluster_id, regla in mejores_reglas.items():
            archivo_texto.write(f"Clúster {cluster_id}:\n{regla}\n\n")
            
    return mejores_reglas

if __name__ == "__main__":
    base_de_datos = generar_base_datos_aleatoria(num_instancias=10, num_objetos=50)
    clasificar_y_evolucionar(base_de_datos, num_clusters=2, generaciones=20)
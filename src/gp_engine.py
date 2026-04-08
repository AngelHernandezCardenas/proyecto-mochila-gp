import operator
import multiprocessing
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
from sklearn.cluster import KMeans
from knaspsack import Item, KnapsackState, KnapsackInstance

def div_segura(izq, der):
    # Prevención de división por cero
    if der == 0:
        return 1.0
    return izq / der

# Definición del conjunto de terminales y operaciones lógicas
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura, 2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')

# Configuración de la estructura evolutiva DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Evaluación de la hiper-heurística con control de memoria (Bloat)
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
    puntaje_final = mochila.current_profit - penalizacion_longitud
    
    return puntaje_final,

# Configuración de operadores genéticos
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Límites estáticos para contener el crecimiento del árbol
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

# Agrupamiento, Evolución Distribuida y Exportación de Datos
def clasificar_y_evolucionar(lista_instancias, num_clusters=2, generaciones=15):
    caracteristicas = []
    
    # Extracción vectorial
    for instancia in lista_instancias:
        promedio_p = np.mean([item.profit for item in instancia.items])
        promedio_w = np.mean([item.weight for item in instancia.items])
        caracteristicas.append([promedio_p, promedio_w])
        
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(caracteristicas)
    
    clusters = {i: [] for i in range(num_clusters)}
    for idx, etiqueta in enumerate(etiquetas):
        clusters[etiqueta].append(lista_instancias[idx])
        
    mejores_reglas = {}
    
    # Activación del procesamiento en paralelo (Multiprocessing)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    for cluster_id, instancias_cluster in clusters.items():
        if not instancias_cluster:
            continue
            
        print(f"\n Iniciando evolución paralela para el Clúster {cluster_id} ")
        poblacion = toolbox.population(n=50)
        instancia_prueba = instancias_cluster[0]
        
        toolbox.register("evaluate", evaluar_hiper_heuristica, instancia=instancia_prueba)
        
        estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0])
        estadisticas.register("Promedio", np.mean)
        estadisticas.register("Max_Ganancia", np.max)
        estadisticas.register("Desviacion", np.std)
        
        salon_fama = tools.HallOfFame(1)
        
        # Bucle generacional automático
        poblacion_final, bitacora = algorithms.eaSimple(
            poblacion, toolbox, cxpb=0.7, mutpb=0.2, ngen=generaciones, 
            stats=estadisticas, halloffame=salon_fama, verbose=True
        )
        
        # Guardado del árbol ganador en memoria
        mejor_individuo = salon_fama[0]
        mejores_reglas[cluster_id] = str(mejor_individuo)
        
        # Automatización de reportes CSV mediante Pandas
        df_log = pd.DataFrame(bitacora)
        nombre_archivo_csv = f"bitacora_cluster_{cluster_id}.csv"
        df_log.to_csv(nombre_archivo_csv, index=False)
        print(f"Datos exportados exitosamente a: {nombre_archivo_csv}")
        
    # Cierre de los hilos de procesamiento
    pool.close()
    pool.join()
    
    # Automatización de guardado de fórmulas ganadoras
    with open("mejores_reglas_historicas.txt", "w") as archivo_texto:
        archivo_texto.write(" REPORTE DE FÓRMULAS EVOLUTIVAS \n")
        for cluster_id, regla in mejores_reglas.items():
            archivo_texto.write(f"Clúster {cluster_id}:\n{regla}\n\n")
    print("\nFórmulas ganadoras respaldadas en: mejores_reglas_historicas.txt")
        
    return mejores_reglas

# Bloque de ejecución principal
if __name__ == "__main__":
    items_instancia_1 = [Item(1,5,10), Item(2,4,40), Item(3, 6, 30), Item(4, 3, 50), Item(5, 2, 15)]
    instancia_1 = KnapsackInstance("Instancia_1", 12, items_instancia_1)
    
    items_instancia_2 = [Item(1, 2, 15), Item(2, 5, 20), Item(3, 8, 25), Item(4, 1, 10), Item(5, 4, 30)]
    instancia_2 = KnapsackInstance("Instancia_2", 10, items_instancia_2)
    
    base_de_datos = [instancia_1, instancia_2]
    
    clasificar_y_evolucionar(base_de_datos, num_clusters=2, generaciones=15)
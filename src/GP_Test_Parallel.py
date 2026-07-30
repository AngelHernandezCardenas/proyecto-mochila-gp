import operator
import random
import numpy as np
import pandas as pd
import itertools
import statistics
import os
import json
import signal
import time
import glob
import psutil
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, gp, algorithms

# 1. BLOQUE DE EXPLORACIÓN DE HIPERPARÁMETROS (TESTING MANUAL)
POBLACION_TAM_LIST = [5, 50, 100]              # Aquí se modifica la población
MUT_PB_LIST        = [0.0, 0.1, 0.5, 0.7, 1.0] # Aquí se modifica la mutación
GEN_LIST           = [5, 20, 50, 100]          # Aquí se modifican las generaciones
CRUCE_PB_LIST      = [0.0, 0.1, 0.5, 0.7, 1.0] # Aquí se modifica el cruce
FASES_LIST         = [5, 20, 50, 100]          # Aquí se modifican las fases (entornos)

REPLICAS_SCORE = 5 # Guardar 5 scores independientes y usar la mediana

# Parámetros fijos
MAX_TREE_HEIGHT = 8
NUM_INSTANCIAS  = 10
NUM_OBJETOS     = 50

# CHECKPOINT
CHECKPOINT_FILE = "checkpoint_grid_search_parallel.json"
CSV_RESULTADOS_FINAL  = "resultados_grid_search_gp_completos_parallel.csv"
LOTE_TAMANO = 500

# DETECCIÓN INTELIGENTE DE HARDWARE
def obtener_nucleos_optimos():
    total_cores = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    print(f"\n--- DETECCIÓN DE HARDWARE ---")
    print(f" CPU Detectada: {total_cores} núcleos lógicos.")
    print(f" RAM Detectada: {ram_gb:.1f} GB.")
    
    # Lógica de núcleos
    if total_cores >= 24:
        cores_permitidos = total_cores - 6
    elif total_cores >= 16:
        cores_permitidos = total_cores - 4
    elif total_cores >= 8:
        cores_permitidos = total_cores - 2
    elif total_cores > 4:
        cores_permitidos = total_cores - 1
    else:
        cores_permitidos = max(1, total_cores - 1)
        
    # Lógica de RAM (Estimando ~150MB por worker con DEAP)
    # Por ejemplo, si tenemos 8GB, dejamos 3GB libres para SO = 5GB utiles -> ~34 workers max.
    # Esta regla previene que un PC con poca RAM sature memoria antes que CPU.
    ram_disponible_para_workers_gb = max(1, ram_gb - 3) 
    max_workers_por_ram = int((ram_disponible_para_workers_gb * 1024) / 150)
    
    workers_finales = min(cores_permitidos, max_workers_por_ram)
    
    print(f" -> Límite por CPU: {cores_permitidos} workers")
    print(f" -> Límite por RAM: {max_workers_por_ram} workers")
    print(f" NÚCLEOS ASIGNADOS A DEAP: {workers_finales}\n")
    return workers_finales


# 2. CLASES DEL PROBLEMA (MOCHILA)
class Item:
    def __init__(self, id_item, weight, profit):
        self.id     = id_item
        self.weight = weight
        self.profit = profit
        self.ratio  = profit / weight

class KnapsackState:
    def __init__(self, capacity):
        self.capacity       = capacity
        self.current_weight = 0.0
        self.current_profit = 0.0

    def pack(self, item):
        if self.current_weight + item.weight <= self.capacity:
            self.current_weight += item.weight
            self.current_profit += item.profit

class KnapsackInstance:
    def __init__(self, id_instancia, capacity, items):
        self.id       = id_instancia
        self.capacity = capacity
        self.items    = items

# 3. CONFIGURACIÓN DEL ÁRBOL GP Y DEAP
def div_segura(izq, der):
    return izq / der if abs(der) > 1e-6 else 1.0

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div_segura,   2)
pset.renameArguments(ARG0='P', ARG1='W', ARG2='PW')

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr",       gp.genHalfAndHalf,  pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate,   creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat,    list, toolbox.individual)
toolbox.register("compile",    gp.compile,          pset=pset)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate",   gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))

# 4. FUNCIONES DE EVALUACIÓN Y ENTORNO
def evaluar_robusto(individuo):
    global CURRENT_INSTANCES
    try:
        rutina_puntuacion = toolbox.compile(expr=individuo)
    except Exception:
        return (-np.inf,)

    ganancias = []
    for instancia in CURRENT_INSTANCES:
        mochila = KnapsackState(capacity=instancia.capacity)
        items_puntuados = []
        for item in instancia.items:
            try:
                score = rutina_puntuacion(item.profit, item.weight, item.ratio)
                items_puntuados.append((score, item))
            except Exception:
                continue
        items_puntuados.sort(key=lambda x: x[0], reverse=True)
        for _, item in items_puntuados:
            mochila.pack(item)
        ganancias.append(mochila.current_profit)

    if not ganancias:
        return (-np.inf,)

    penalizacion = len(individuo) * 0.01
    return (np.mean(ganancias) - penalizacion,)

def generar_base_datos_aleatoria(num_instancias=NUM_INSTANCIAS, num_objetos=NUM_OBJETOS):
    instancias = []
    for i in range(num_instancias):
        capacidad = random.uniform(50.0, 150.0)
        objetos   = [Item(j, random.uniform(1.0, 20.0), random.uniform(10.0, 100.0))
                     for j in range(num_objetos)]
        instancias.append(KnapsackInstance(f"Inst_{i}", capacidad, objetos))
    return instancias


# 5. MOTOR EVOLUTIVO ADAPTADO
def clasificar_y_evolucionar(tam_poblacion, generaciones, cxpb, mutpb, elite_anterior=None):
    if elite_anterior:
        poblacion = [toolbox.clone(ind) for ind in elite_anterior]
        while len(poblacion) < tam_poblacion:
            poblacion.append(toolbox.individual())
    else:
        poblacion = toolbox.population(n=tam_poblacion)

    for ind in poblacion:
        del ind.fitness.values

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
    estadisticas.register("Max_Ganancia", np.max)
    
    tam_elite_local = max(1, int(tam_poblacion * 0.1)) 
    salon_fama = tools.HallOfFame(tam_elite_local)

    algorithms.eaSimple(
        poblacion, toolbox,
        cxpb=cxpb, mutpb=mutpb,
        ngen=generaciones,
        stats=estadisticas,
        halloffame=salon_fama,
        verbose=False
    )
    
    mejor_score = salon_fama[0].fitness.values[0]
    return list(salon_fama), mejor_score

# 6. FUNCIONES DE CHECKPOINT
def cargar_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f" Checkpoint cargado: {len(checkpoint['resultados_completados'])} combinaciones procesadas")
            return checkpoint
        except Exception:
            return None
    return None

def guardar_checkpoint(indices_completados, resultados_totales):
    checkpoint = {
        "indices_completados": indices_completados,
        "resultados_totales": resultados_totales
    }
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        print(f" Error al guardar checkpoint: {e}")

def guardar_csv_lote(resultados_lote, lote_id):
    if not resultados_lote: return
    try:
        df = pd.DataFrame(resultados_lote)
        nombre_archivo = f"resultados_GP_parallel_parte_{lote_id}.csv"
        df.to_csv(nombre_archivo, index=False)
        print(f" Lote {lote_id} guardado en {nombre_archivo}")
    except Exception as e:
        print(f" Error al guardar CSV del lote: {e}")

def fusionar_csvs():
    archivos = sorted(glob.glob("resultados_GP_parallel_parte_*.csv"))
    if archivos:
        dfs = [pd.read_csv(f) for f in archivos]
        df_final = pd.concat(dfs, ignore_index=True)
        df_final.to_csv(CSV_RESULTADOS_FINAL, index=False)
        print(f"\n Se fusionaron {len(archivos)} lotes en '{CSV_RESULTADOS_FINAL}'")
        return df_final
    return pd.DataFrame()


# VARIABLE GLOBAL PARA INSTANCIAS (necesario para Multiprocessing)
CURRENT_INSTANCES = []

def inicializar_worker(instancias_fase):
    global CURRENT_INSTANCES
    CURRENT_INSTANCES = instancias_fase

# 7. ORQUESTADOR DE GRID SEARCH CON MULTIPROCESSING
def ejecutar_grid_search_paralelo():
    combinaciones = list(itertools.product(
        POBLACION_TAM_LIST, MUT_PB_LIST, GEN_LIST, CRUCE_PB_LIST, FASES_LIST
    ))
    
    checkpoint_anterior = cargar_checkpoint()
    if checkpoint_anterior:
        indices_completados = set(checkpoint_anterior["indices_completados"])
        resultados_totales = checkpoint_anterior["resultados_totales"]
        print(f"ℹ️  Reanudando desde índice {max(indices_completados) + 1 if indices_completados else 0}\n")
    else:
        indices_completados = set()
        resultados_totales = []
    
    lote_actual = (len(indices_completados) // LOTE_TAMANO) + 1
    resultados_lote_actual = []
    if len(indices_completados) % LOTE_TAMANO != 0:
        resultados_lote_actual = resultados_totales[-(len(indices_completados) % LOTE_TAMANO):]
    
    total_combinaciones = len(combinaciones)
    print(f"Iniciando Grid Search Paralelo. Total de combinaciones: {total_combinaciones}")
    
    # Asignar evaluación al toolbox
    toolbox.register("evaluate", evaluar_robusto)
    
    try:
        for idx, (P_tam, Mut, Gen, Cx, Fases) in enumerate(combinaciones):
            if idx in indices_completados:
                continue
            
            print(f"[{idx+1}/{total_combinaciones}] Pob: {P_tam}, Mut: {Mut}, Gen: {Gen}, Cx: {Cx}, Fases: {Fases}")
            
            try:
                tiempo_inicio = time.time()
                scores_replicas = []
                
                # Averiguar núcleos
                workers = obtener_nucleos_optimos()
                
                for replica in range(REPLICAS_SCORE):
                    pool_elite = None
                    score_final_replica = 0
                    
                    for f in range(Fases):
                        base_de_datos = generar_base_datos_aleatoria()
                        
                        # Iniciar pool de procesamiento para esta fase
                        with multiprocessing.Pool(processes=workers, initializer=inicializar_worker, initargs=(base_de_datos,)) as pool:
                            toolbox.register("map", pool.map)
                            
                            pool_elite, score_fase = clasificar_y_evolucionar(
                                tam_poblacion=P_tam,
                                generaciones=Gen,
                                cxpb=Cx,
                                mutpb=Mut,
                                elite_anterior=pool_elite
                            )
                            
                            # Limpiar map para no colisionar en siguiente iter
                            toolbox.unregister("map")
                            
                        score_final_replica = score_fase
                        
                    scores_replicas.append(score_final_replica)
                
                mediana_score = statistics.median(scores_replicas)
                tiempo_fin = time.time()
                tiempo_segundos = tiempo_fin - tiempo_inicio
                
                resultado = {
                    "Poblacion": P_tam,
                    "Mutacion_Pb": Mut,
                    "Generaciones": Gen,
                    "Cruce_Pb": Cx,
                    "Num_Fases": Fases,
                    "Score_Rep1": scores_replicas[0],
                    "Score_Rep2": scores_replicas[1],
                    "Score_Rep3": scores_replicas[2],
                    "Score_Rep4": scores_replicas[3],
                    "Score_Rep5": scores_replicas[4],
                    "Mediana_Score": mediana_score,
                    "Tiempo_Inicio": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tiempo_inicio)),
                    "Tiempo_Fin": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tiempo_fin)),
                    "Tiempo_Segundos": round(tiempo_segundos, 2)
                }
                resultados_totales.append(resultado)
                resultados_lote_actual.append(resultado)
                indices_completados.add(idx)
                
                print(f"     Mediana: {mediana_score:.2f} | Tiempo: {tiempo_segundos:.2f}s")
                
                guardar_checkpoint(list(indices_completados), resultados_totales)
                
                if len(indices_completados) % LOTE_TAMANO == 0:
                    guardar_csv_lote(resultados_lote_actual, lote_actual)
                    resultados_lote_actual = []
                    lote_actual += 1
                
            except Exception as e:
                print(f"  Error en combinación {idx}: {e}")
                import traceback
                traceback.print_exc()
                guardar_checkpoint(list(indices_completados), resultados_totales)
                continue
        
        if resultados_lote_actual:
            guardar_csv_lote(resultados_lote_actual, lote_actual)
            
        df_resultados = fusionar_csvs()
        print("\n Grid Search Paralelo completado exitosamente")
        
    except KeyboardInterrupt:
        print("\n Checkpoint guardado. Para reanudar, ejecuta el script nuevamente.")
        if resultados_lote_actual:
            guardar_csv_lote(resultados_lote_actual, lote_actual)
    
    df_resultados = fusionar_csvs()
    if df_resultados.empty:
        df_resultados = pd.DataFrame(resultados_totales)
        
    return df_resultados

if __name__ == "__main__":
    multiprocessing.freeze_support() # Necesario en Windows
    df_final = ejecutar_grid_search_paralelo()
    print(f"\nResultados paralelos disponibles en '{CSV_RESULTADOS_FINAL}'")
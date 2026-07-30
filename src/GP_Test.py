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
CHECKPOINT_FILE = "checkpoint_grid_search.json"
CSV_RESULTADOS_FINAL  = "resultados_grid_search_gp_completos.csv"
LOTE_TAMANO = 500

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

# Maximization (Verificación solicitada)
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
def evaluar_robusto(individuo, lista_instancias):
    try:
        rutina_puntuacion = toolbox.compile(expr=individuo)
    except Exception:
        return (-np.inf,)

    ganancias = []
    for instancia in lista_instancias:
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


# 5. MOTOR EVOLUTIVO ADAPTADO PARA RECIBIR PARÁMETROS DINÁMICOS
def clasificar_y_evolucionar(lista_instancias, tam_poblacion, generaciones, cxpb, mutpb, elite_anterior=None):
    if elite_anterior:
        poblacion = [toolbox.clone(ind) for ind in elite_anterior]
        while len(poblacion) < tam_poblacion:
            poblacion.append(toolbox.individual())
    else:
        poblacion = toolbox.population(n=tam_poblacion)

    toolbox.register("evaluate", evaluar_robusto, lista_instancias=lista_instancias)

    for ind in poblacion:
        del ind.fitness.values

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
    estadisticas.register("Max_Ganancia", np.max)
    
    # Conservar top 10% como élite
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
    """Carga el checkpoint si existe."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f" Checkpoint cargado: {len(checkpoint['resultados_completados'])} combinaciones procesadas")
            return checkpoint
        except Exception as e:
            print(f" Error al cargar checkpoint: {e}")
            return None
    return None

def guardar_checkpoint(indices_completados, resultados_totales):
    """Guarda el checkpoint actual."""
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
    """Guarda los resultados de un lote en CSV."""
    if not resultados_lote: return
    try:
        df = pd.DataFrame(resultados_lote)
        nombre_archivo = f"resultados_GP_parte_{lote_id}.csv"
        df.to_csv(nombre_archivo, index=False)
        print(f" Lote {lote_id} guardado en {nombre_archivo}")
    except Exception as e:
        print(f" Error al guardar CSV del lote: {e}")

def fusionar_csvs():
    """Une todos los lotes guardados en un solo archivo."""
    archivos = sorted(glob.glob("resultados_GP_parte_*.csv"))
    if archivos:
        dfs = [pd.read_csv(f) for f in archivos]
        df_final = pd.concat(dfs, ignore_index=True)
        df_final.to_csv(CSV_RESULTADOS_FINAL, index=False)
        print(f"\n Se fusionaron {len(archivos)} lotes en '{CSV_RESULTADOS_FINAL}'")
        return df_final
    return pd.DataFrame()


# 7. ORQUESTADOR DE GRID SEARCH CON CHECKPOINTS
class GestorInterrupcion:
    def __init__(self):
        self.interrumpido = False
        signal.signal(signal.SIGINT, self._manejar_interrupcion)
        signal.signal(signal.SIGTERM, self._manejar_interrupcion)
    
    def _manejar_interrupcion(self, signum, frame):
        print("\n\n  INTERRUPCIÓN DETECTADA (Ctrl+C o error crítico)")
        print(" Guardando checkpoint antes de salir...")
        self.interrumpido = True
        raise KeyboardInterrupt("Interrupción del usuario")

gestor_interrupcion = GestorInterrupcion()

def ejecutar_grid_search():
    combinaciones = list(itertools.product(
        POBLACION_TAM_LIST, MUT_PB_LIST, GEN_LIST, CRUCE_PB_LIST, FASES_LIST
    ))
    
    # Verificar si existe checkpoint anterior
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
        # Recuperar resultados parciales del lote actual
        resultados_lote_actual = resultados_totales[-(len(indices_completados) % LOTE_TAMANO):]
    
    total_combinaciones = len(combinaciones)
    print(f"Iniciando Grid Search. Total de combinaciones: {total_combinaciones}")
    print(f"Completadas: {len(indices_completados)}, Faltantes: {total_combinaciones - len(indices_completados)}\n")
    
    try:
        for idx, (P_tam, Mut, Gen, Cx, Fases) in enumerate(combinaciones):
            # Saltar si ya está completada
            if idx in indices_completados:
                continue
            
            print(f"[{idx+1}/{total_combinaciones}] Probando -> Pob: {P_tam}, Mut: {Mut}, Gen: {Gen}, Cx: {Cx}, Fases: {Fases}")
            
            try:
                tiempo_inicio = time.time()
                scores_replicas = []
                
                # Correr réplicas
                for replica in range(REPLICAS_SCORE):
                    pool_elite = None
                    score_final_replica = 0
                    
                    # Recorrer fases de la réplica
                    for f in range(Fases):
                        base_de_datos = generar_base_datos_aleatoria()
                        pool_elite, score_fase = clasificar_y_evolucionar(
                            lista_instancias=base_de_datos,
                            tam_poblacion=P_tam,
                            generaciones=Gen,
                            cxpb=Cx,
                            mutpb=Mut,
                            elite_anterior=pool_elite
                        )
                        score_final_replica = score_fase
                        
                    scores_replicas.append(score_final_replica)
                
                mediana_score = statistics.median(scores_replicas)
                tiempo_fin = time.time()
                tiempo_segundos = tiempo_fin - tiempo_inicio
                
                # Guardar resultado
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
                
                # Guardar checkpoint cada iteración
                guardar_checkpoint(list(indices_completados), resultados_totales)
                
                if len(indices_completados) % LOTE_TAMANO == 0:
                    guardar_csv_lote(resultados_lote_actual, lote_actual)
                    resultados_lote_actual = []
                    lote_actual += 1
                
            except Exception as e:
                print(f"  Error en combinación {idx}: {e}")
                # Guardar antes de continuar
                guardar_checkpoint(list(indices_completados), resultados_totales)
                # No guardamos csv por lote en caso de error medio lote para no romper la secuencia normal
                continue
        
        # Al finalizar, guardar el lote remanente
        if resultados_lote_actual:
            guardar_csv_lote(resultados_lote_actual, lote_actual)
            
        # Fusionar todos los lotes
        df_resultados = fusionar_csvs()
        
        print("\n Grid Search completado exitosamente")
        
        # Limpiar checkpoint al terminar
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print(" Checkpoint limpiado")
        
    except KeyboardInterrupt:
        print("\n Checkpoint guardado. Para reanudar, ejecuta el script nuevamente.")
        print(f"   Progreso: {len(indices_completados)}/{total_combinaciones}")
        if resultados_lote_actual:
            guardar_csv_lote(resultados_lote_actual, lote_actual)
    
    # En caso de interrupción intentamos fusionar lo que haya
    df_resultados = fusionar_csvs()
    if df_resultados.empty:
        df_resultados = pd.DataFrame(resultados_totales)
        
    return df_resultados

def graficar_comportamiento_e_interacciones(df):
    """
    Check interacciones: Genera gráficos para visualizar cómo afectan 
    los parámetros al score (Mediana).
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Gráfico de pares (Pairplot) para ver correlaciones generales
    columnas_interes = ["Poblacion", "Mutacion_Pb", "Generaciones", "Cruce_Pb", "Num_Fases", "Mediana_Score"]
    df_interes = df[columnas_interes]
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(df_interes, y_vars=["Mediana_Score"], x_vars=columnas_interes[:-1], kind="reg", height=4)
    plt.suptitle("Check de Interacciones: Impacto de Parámetros en la Mediana del Score", y=1.05)
    plt.savefig("src/graficas/interacciones_parametros.png", bbox_inches='tight')
    plt.close()
    print(" Gráfico de interacciones guardado: interacciones_parametros.png")
    
    # 2. Matriz de Correlación
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_interes.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación de Parámetros")
    plt.savefig("src/graficas/matriz_correlacion.png", bbox_inches='tight')
    plt.close()
    print(" Matriz de correlación guardada: matriz_correlacion.png")

if __name__ == "__main__":
    # 1. Ejecutar la búsqueda de hiperparámetros
    df_final = ejecutar_grid_search()
    
    # 2. Informar sobre resultados
    print(f"\nResultados completos disponibles en '{CSV_RESULTADOS_FINAL}'")
    
    # 3. Graficar el comportamiento final
    if len(df_final) > 1:
        print("\nGenerando gráficas de interacciones...")
        graficar_comportamiento_e_interacciones(df_final)
    else:
        print("\nSolo hay 1 combinación probada. Modificar las listas arriba para generar gráficas comparativas.")

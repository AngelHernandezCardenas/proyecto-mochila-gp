import pandas as pd
import io

def generar_dataset_weka(ruta_entrada, ruta_salida):
    """
    Lee el archivo de texto con todas las fases y genera un único archivo CSV 
    limpio y estructurado, listo para importar a WEKA.
    """
    fases = {}
    fase_actual = None
    datos_fase = []
    
    # 1. Leer y estructurar los datos (misma lógica robusta)
    with open(ruta_entrada, 'r', encoding='utf-8-sig') as file:
        for linea in file:
            linea = linea.strip()
            if not linea:
                continue
                
            if linea.lower().startswith("fase"):
                if fase_actual and datos_fase:
                    fases[fase_actual] = pd.read_csv(io.StringIO('\n'.join(datos_fase)))
                
                fase_actual = linea.lower().replace("fase", "").strip()
                datos_fase = []
                
            elif linea == "gen,nevals,Promedio,Max_Ganancia,Desviacion":
                if len(datos_fase) == 0:
                    datos_fase.append(linea)
            else:
                datos_fase.append(linea)
                
        if fase_actual and datos_fase:
            fases[fase_actual] = pd.read_csv(io.StringIO('\n'.join(datos_fase)))

    # 2. Consolidar en un solo DataFrame con la nueva columna para WEKA
    df_consolidado = pd.DataFrame()
    
    for numero_fase, df in fases.items():
        df_temp = df.copy()
        # Agregamos el prefijo 'Fase_' para que WEKA lo detecte como clase Nominal y no numérica
        df_temp['Clase_Fase'] = f"Fase_{numero_fase}" 
        df_consolidado = pd.concat([df_consolidado, df_temp], ignore_index=True)
    
    # 3. Exportar al CSV final
    df_consolidado.to_csv(ruta_salida, index=False)
    print(f"¡Éxito! Se procesaron {len(fases)} fases.")
    print(f"El archivo para WEKA se guardó en: {ruta_salida}")

if __name__ == "__main__":
    # Rutas absolutas para evitar problemas de directorios
    archivo_entrada = r"C:\Users\jrhe0\proyecto-mochila-gp\src\Todas_las_bitacoras.txt"
    archivo_salida_weka = r"C:\Users\jrhe0\proyecto-mochila-gp\src\bitacoras_para_weka.csv"
    
    try:
        generar_dataset_weka(archivo_entrada, archivo_salida_weka)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo_entrada}'.")
import pandas as pd
import matplotlib.pyplot as plt
import io

def leer_y_separar_fases(ruta_archivo):
    """
    Lee el archivo de texto que contiene todas las fases y las separa 
    en un diccionario de DataFrames de pandas.
    """
    fases = {}
    fase_actual = None
    datos_fase = []
    
    with open(ruta_archivo, 'r') as file:
        for linea in file:
            linea = linea.strip()
            if not linea: # <-- ¡Aquí estaba el error! (Cambiado de 'no' a 'not')
                continue
                
            if linea.startswith("Fase "):
                # Guardar la fase anterior si existe
                if fase_actual and datos_fase:
                    fases[fase_actual] = pd.read_csv(io.StringIO('\n'.join(datos_fase)))
                
                fase_actual = linea.replace("Fase ", "").strip()
                datos_fase = []
            elif linea == "gen,nevals,Promedio,Max_Ganancia,Desviacion":
                # Asegurarse de no duplicar la cabecera si viene repetida
                if len(datos_fase) == 0:
                    datos_fase.append(linea)
            else:
                datos_fase.append(linea)
                
        # Guardar la última fase
        if fase_actual and datos_fase:
            fases[fase_actual] = pd.read_csv(io.StringIO('\n'.join(datos_fase)))
            
    return fases

def graficar_convergencia(df, numero_fase):
    """
    Genera la gráfica de convergencia evolutiva para un DataFrame específico.
    """
    plt.figure(figsize=(12, 7))
    
    # Extraer columnas
    generaciones = df['gen']
    promedio = df['Promedio']
    max_ganancia = df['Max_Ganancia']
    desviacion = df['Desviacion']
    
    # Rellenar el área de la Desviación Estándar (Caos Genético)
    plt.fill_between(
        generaciones, 
        promedio - desviacion, 
        promedio + desviacion, 
        color='lightgreen', 
        alpha=0.3, 
        label='Desviación Estándar (Caos Genético)'
    )
    
    # Línea principal del Promedio
    plt.plot(
        generaciones, 
        promedio, 
        marker='o', 
        color='forestgreen', 
        linewidth=2, 
        label='Aptitud Promedio de la Población'
    )
    
    # Línea de la Ganancia Máxima
    plt.plot(
        generaciones, 
        max_ganancia, 
        linestyle='--', 
        color='black', 
        linewidth=2, 
        label='Límite de Ganancia Máxima (Elitismo)'
    )
    
    # Configuración de diseño y etiquetas
    plt.title(f'Convergencia Evolutiva y Reducción del Caos (Fase {numero_fase})', fontsize=14, fontweight='bold')
    plt.xlabel('Generaciones Evolutivas', fontsize=12)
    plt.ylabel('Ganancia Matemática (Fitness)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Ajustar límites del eje X
    plt.xlim(0, max(generaciones))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Actualizado con el nombre exacto de tu archivo
    archivo_datos = r"C:\Users\jrhe0\proyecto-mochila-gp\src\Todas_las_bitacoras.txt" 
    
    try:
        # 1. Leer y estructurar los datos
        diccionario_fases = leer_y_separar_fases(archivo_datos)
        print(f"Se cargaron exitosamente {len(diccionario_fases)} fases.")
        
        # 2. Elegir qué fase graficar
        fase_a_graficar = '5' 
        
        if fase_a_graficar in diccionario_fases:
            df_fase = diccionario_fases[fase_a_graficar]
            graficar_convergencia(df_fase, fase_a_graficar)
        else:
            print(f"La Fase {fase_a_graficar} no se encontró en los datos.")
            
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo_datos}'. Asegúrate de que esté en la misma carpeta que este script.")
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def generar_pareto_front(df, x_col, y_col):
    """
    Encuentra los puntos del frente de Pareto asumiendo que:
    x_col (Tiempo) se debe MINIMIZAR
    y_col (Score) se debe MAXIMIZAR
    """
    # Ordenamos por x_col (ascendente) y luego y_col (descendente)
    puntos_ordenados = df.sort_values(by=[x_col, y_col], ascending=[True, False]).copy()
    
    pareto_front = []
    max_y_encontrado = -float('inf')
    
    for index, row in puntos_ordenados.iterrows():
        if row[y_col] > max_y_encontrado:
            pareto_front.append(index)
            max_y_encontrado = row[y_col]
            
    return puntos_ordenados.loc[pareto_front]

def analizar_resultados(ruta_csv):
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encuentra el archivo {ruta_csv}")
        return
    
    df = pd.read_csv(ruta_csv)
    print(f"Datos cargados: {df.shape[0]} configuraciones probadas.")
    
    # 1. Preparar datos para ANOVA (Melt de réplicas)
    columnas_params = ['Poblacion', 'Mutacion_Pb', 'Generaciones', 'Cruce_Pb', 'Num_Fases']
    columnas_scores = ['Score_Rep1', 'Score_Rep2', 'Score_Rep3', 'Score_Rep4', 'Score_Rep5']
    
    # Crear un identificador único de configuración
    df['Config'] = df[columnas_params].apply(lambda row: f"P{int(row['Poblacion'])}_M{row['Mutacion_Pb']}_G{int(row['Generaciones'])}_C{row['Cruce_Pb']}_F{int(row['Num_Fases'])}", axis=1)
    
    df_melt = pd.melt(df, id_vars=['Config'], value_vars=columnas_scores, var_name='Replica', value_name='Score')
    
    # 2. ANOVA
    print("\n--- Realizando ANOVA ---")
    grupos = [grupo['Score'].values for name, grupo in df_melt.groupby('Config')]
    f_stat, p_val = stats.f_oneway(*grupos)
    print(f"Estadístico F: {f_stat:.4f}, Valor p: {p_val:.4e}")
    
    if p_val < 0.05:
        print("Hay diferencias significativas entre al menos dos configuraciones (p < 0.05).")
    else:
        print("No se encontraron diferencias significativas entre las configuraciones.")
    
    # 3. Tukey HSD
    print("\n--- Realizando Prueba Tukey ---")
    # Para no saturar memoria si hay miles de grupos, vamos a filtrar las mejores N configuraciones
    # si hay demasiadas para el análisis completo.
    TOP_N = 30
    mejores_configs = df.sort_values(by='Mediana_Score', ascending=False).head(TOP_N)['Config']
    
    df_melt_top = df_melt[df_melt['Config'].isin(mejores_configs)].copy()
    tukey = pairwise_tukeyhsd(endog=df_melt_top['Score'], groups=df_melt_top['Config'], alpha=0.05)
    
    print(f"Se analizó el top {TOP_N} configuraciones con Tukey para encontrar diferencias.")
    # Extraer el mejor grupo absoluto
    mejor_config_nombre = mejores_configs.iloc[0]
    print(f"Mejor configuración absoluta (por mediana): {mejor_config_nombre}")
    
    # Convertir resultados de Tukey a DataFrame para buscar similares
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    
    # Buscar grupos que comparados con el mejor tengan reject == False (estadísticamente similares)
    similares_al_mejor = set([mejor_config_nombre])
    for index, row in tukey_df.iterrows():
        if row['group1'] == mejor_config_nombre and not row['reject']:
            similares_al_mejor.add(row['group2'])
        elif row['group2'] == mejor_config_nombre and not row['reject']:
            similares_al_mejor.add(row['group1'])
            
    print(f"\nConfiguraciones estadísticamente SIMILARES a la mejor ({len(similares_al_mejor)} encontradas):")
    for sim in similares_al_mejor:
        print(f" - {sim}")
        
    # 4. Gráfica Boxplot orientada a la derecha
    sns.set_theme(style="whitegrid")
    # Ordenar por Mediana_Score (usando el df original)
    orden_configs = df[df['Config'].isin(similares_al_mejor)].sort_values('Mediana_Score', ascending=False)['Config'].tolist()
    
    # Altura dinámica según cantidad de barras para que el texto no se amontone
    plt.figure(figsize=(12, max(8, len(orden_configs) * 0.4)))
    
    # Filtramos df_melt solo para las estadisticamente similares para dibujarlas
    df_plot_similares = df_melt[df_melt['Config'].isin(similares_al_mejor)]
    
    # color uniforme en vez de palette, y ordenadas de mejor a peor
    sns.boxplot(data=df_plot_similares, x='Score', y='Config', order=orden_configs, orient='h', color='#85C1E9')
    plt.title('Boxplot de Configuraciones Estadísticamente Similares a la Mejor', fontsize=14)
    plt.xlabel('Score (Ganancia)', fontsize=12)
    plt.ylabel('Configuración', fontsize=12)
    plt.tight_layout()
    plt.savefig('src/graficas/boxplot_mejores_configs.png', dpi=300)
    plt.close()
    print("\n Gráfica Boxplot guardada como 'src/graficas/boxplot_mejores_configs.png'")
    
    # 5. Análisis de Frente de Pareto (Tiempo vs Quality)
    print("\n--- Análisis Frente de Pareto ---")
    if 'Tiempo_Segundos' in df.columns:
        pareto_df = generar_pareto_front(df, x_col='Tiempo_Segundos', y_col='Mediana_Score')
        print(f"Se encontraron {len(pareto_df)} configuraciones en el frente de Pareto.")
        
        plt.figure(figsize=(10, 6))
        # Puntos generales
        plt.scatter(df['Tiempo_Segundos'], df['Mediana_Score'], color='gray', alpha=0.5, label='Otras Configs')
        # Puntos Pareto
        plt.scatter(pareto_df['Tiempo_Segundos'], pareto_df['Mediana_Score'], color='red', marker='D', s=50, label='Frente de Pareto')
        
        # Conectar los puntos de Pareto con una línea
        plt.plot(pareto_df['Tiempo_Segundos'], pareto_df['Mediana_Score'], color='red', linestyle='--')
        
        plt.title('Trade-off (Tiempo vs Calidad) y Frente de Pareto', fontsize=14)
        plt.xlabel('Tiempo de Ejecución (segundos) - Minimizar', fontsize=12)
        plt.ylabel('Mediana del Score - Maximizar', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig('src/graficas/pareto_front.png', dpi=300)
        plt.close()
        print(" Gráfica del Frente de Pareto guardada como 'src/graficas/pareto_front.png'")
        
        print("\nConfiguraciones en el frente de Pareto (Balance Tiempo/Calidad):")
        print(pareto_df[['Config', 'Tiempo_Segundos', 'Mediana_Score']].to_string(index=False))
    else:
        print("No se encontró la columna 'Tiempo_Segundos'. Asegúrate de haber corrido las pruebas con los nuevos cambios para incluir tiempos.")

if __name__ == "__main__":
    ruta_datos = "resultados_grid_search_gp_completos.csv"
    analizar_resultados(ruta_datos)

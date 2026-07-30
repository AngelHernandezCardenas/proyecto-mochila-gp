import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_pareto_front(df, x_col, y_col, minimize_x=True, maximize_y=True):
    # Sort data
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[minimize_x, not maximize_y])
    
    pareto_front = []
    best_y = -np.inf if maximize_y else np.inf
    
    for index, row in sorted_df.iterrows():
        current_y = row[y_col]
        
        if maximize_y:
            if current_y > best_y:
                pareto_front.append(row)
                best_y = current_y
        else:
            if current_y < best_y:
                pareto_front.append(row)
                best_y = current_y
                
    return pd.DataFrame(pareto_front)

def main():
    try:
        df = pd.read_csv('resultados_grid_search_gp_completos.csv')
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        return
        
    print(f"Datos cargados: {len(df)} configuraciones.")
    
    if 'Tiempo_Segundos' not in df.columns:
        print("No se encuentra la columna 'Tiempo_Segundos' para el análisis.")
        return
        
    pareto_df = get_pareto_front(df, 'Tiempo_Segundos', 'Mediana_Score', minimize_x=True, maximize_y=True)
    pareto_df = pareto_df.sort_values(by='Tiempo_Segundos')
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df['Tiempo_Segundos'], df['Mediana_Score'], 
                color='lightgray', alpha=0.5, label='Todas las Configuraciones', s=10)
    
    plt.plot(pareto_df['Tiempo_Segundos'], pareto_df['Mediana_Score'], 
             color='red', marker='o', linestyle='-', linewidth=2, markersize=8, label='Frente de Pareto (Óptimos)')
    
    if len(pareto_df) <= 10:
        for index, row in pareto_df.iterrows():
            config_name = row['Config'] if 'Config' in df.columns else f"P{int(row['Poblacion'])}_G{int(row['Generaciones'])}"
            plt.annotate(config_name, 
                         (row['Tiempo_Segundos'], row['Mediana_Score']),
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='darkred')

    plt.title('Curva del Frente de Pareto (Tiempo vs Ganancia)', fontsize=14)
    plt.xlabel('Tiempo Promedio de Ejecución (Segundos) - Menos es Mejor', fontsize=12)
    plt.ylabel('Score (Ganancia Mediana) - Más es Mejor', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('src/graficas/pareto_curva.png', dpi=300)
    plt.close()
    
    print("\n Gráfica del Frente de Pareto guardada exitosamente como 'src/graficas/pareto_curva.png'")
    print(f"Puntos en el frente: {len(pareto_df)}")

if __name__ == '__main__':
    main()

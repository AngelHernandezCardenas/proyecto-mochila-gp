import random
import matplotlib.pyplot as plt
import numpy as np

def generar_datos_correlacion(num_objetos=200):
    pesos = []
    ganancias = []
    ratios = []
  #simulación de todos los datos obtenidos con respecto al promedio de la iteraciones.   
    for _ in range(num_objetos):
        w = random.uniform(1.0, 20.0)
        p = random.uniform(10.0, 100.0)
        pesos.append(w)
        ganancias.append(p)
        ratios.append(p / w)
        
    return pesos, ganancias, ratios

def graficar_correlacion():
    pesos, ganancias, ratios = generar_datos_correlacion(200)
    
    matriz_corr = np.corrcoef(pesos, ganancias)
    correlacion_pearson = matriz_corr[0, 1]
    
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(pesos, ganancias, c=ratios, cmap='viridis', alpha=0.8, edgecolors='w', s=80)
    
    z = np.polyfit(pesos, ganancias, 1)
    p = np.poly1d(z)
    plt.plot(pesos, p(pesos), "r--", alpha=0.7, label=f'Tendencia lineal (R = {correlacion_pearson:.3f})')
    
    plt.title('Diagrama de Correlacion: Peso vs Ganancia (Espacio de Busqueda)', fontsize=14, pad=15)
    plt.xlabel('Peso Físico del Objeto (W)', fontsize=12)
    plt.ylabel('Ganancia Financiera (P)', fontsize=12)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Aptitud Heuristica (Ratio P/W)', rotation=270, labelpad=15)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.savefig('correlacion.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    graficar_correlacion()
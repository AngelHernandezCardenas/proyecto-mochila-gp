import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ruta del archivo de datos (relativa a la ubicación de este script)
directorio_actual = os.path.dirname(os.path.abspath(__file__))
archivo_datos = os.path.join(directorio_actual, 'ResultadosGP.xlsx')
df = pd.read_excel(archivo_datos)

# Configurar el estilo visual de las gráficas (opcional pero se ve más profesional)
sns.set_theme(style="whitegrid")

# GRÁFICA 1: Mapa de calor (Heatmap)
# Muestra la combinación exacta de Fases y Cruce que da el mejor Score
plt.figure(figsize=(10, 6))

# Creamos una tabla pivote relacionando Cruce_Pb y Num_Fases con la Mediana_Score
pivot_table = df.pivot_table(values='Mediana_Score', 
                             index='Cruce_Pb', 
                             columns='Num_Fases', 
                             aggfunc='mean')

# Generamos el mapa de calor
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor: Impacto de Probabilidad de Cruce y Fases en el Score', fontsize=14)
plt.xlabel('Número de Fases (Num_Fases)', fontsize=12)
plt.ylabel('Probabilidad de Cruce (Cruce_Pb)', fontsize=12)

# Guardar la gráfica como imagen (opcional)
# plt.savefig('mapa_de_calor.png', dpi=300, bbox_inches='tight')
plt.show()

# GRÁFICA 2: Gráfico de líneas con marcadores
# Compara cómo evoluciona el score al aumentar las fases, separado por probabilidad de cruce
plt.figure(figsize=(10, 6))

sns.lineplot(data=df, 
             x='Num_Fases', 
             y='Mediana_Score', 
             hue='Cruce_Pb', 
             marker='o',
             palette='tab10')

plt.title('Evolución del Score Mediano según Número de Fases', fontsize=14)
plt.xlabel('Número de Fases (Num_Fases)', fontsize=12)
plt.ylabel('Score Mediano (Mediana_Score)', fontsize=12)
plt.legend(title='Prob. Cruce (Cruce_Pb)')

# Guardar la gráfica como imagen (opcional)
# plt.savefig('grafico_lineas.png', dpi=300, bbox_inches='tight')
plt.show()
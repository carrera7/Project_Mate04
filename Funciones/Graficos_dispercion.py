import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el CSV con las columnas seleccionadas
df = pd.read_csv("CSV/wdbc_selected.csv")

# Variables
respuesta = 'radio_promedio'
predictoras = ['area_promedio', 'perimetro_promedio', 'concavidad_promedio']

# 1. Gráficos de dispersión individuales
for col in predictoras:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=col, y=respuesta)
    plt.title(f'Dispersión: {col} vs {respuesta}')
    plt.xlabel(col)
    plt.ylabel(respuesta)
    plt.grid(True)
    plt.show()

# 2. Gráfico combinando las variables predictoras
plt.figure(figsize=(8,6))
for col in predictoras:
    plt.scatter(df[col], df[respuesta], label=col, alpha=0.7)
plt.title('Dispersión de variables predictoras vs area_promedio')
plt.xlabel('Valor de la variable predictora')
plt.ylabel(respuesta)
plt.legend()
plt.grid(True)
plt.show()

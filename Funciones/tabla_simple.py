import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ===============================
# Configuración
# ===============================
archivo_csv = "CSV/wdbc_full.csv"  
respuesta = "radio_promedio"
predictora = "perimetro_promedio"

# ===============================
# Cargar datos
# ===============================
df = pd.read_csv(archivo_csv)
X = df[predictora].values
Y = df[respuesta].values

# ===============================
# Cálculo de coeficientes de regresión
# ===============================
n = len(X)
x_mean = np.mean(X)
y_mean = np.mean(Y)

b1 = np.sum((X - x_mean)*(Y - y_mean)) / np.sum((X - x_mean)**2)
b0 = y_mean - b1*x_mean

# ===============================
# Errores y estadísticos globales
# ===============================
Y_hat = b0 + b1*X
residuos = Y - Y_hat
SSE = np.sum(residuos**2)
MSE = SSE/(n-2)
varianza = MSE
Sxx = np.sum((X - x_mean)**2)

# Intervalos de confianza para coeficientes
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, n-2)
se_b1 = np.sqrt(MSE / Sxx)
se_b0 = np.sqrt(MSE * (1/n + x_mean**2 / Sxx))

IC_b1 = (b1 - t_crit*se_b1, b1 + t_crit*se_b1)
IC_b0 = (b0 - t_crit*se_b0, b0 + t_crit*se_b0)

# R² y r
R2 = 1 - SSE/np.sum((Y - y_mean)**2)
r = np.sqrt(R2)

# ===============================
# Seleccionar los primeros 6 valores de X
# ===============================
X_prueba = X[:10]  # 6 filas

# ===============================
# Calcular para cada valor de X_prueba
# ===============================
resultados = []
for x0 in X_prueba:
    y_pred = b0 + b1*x0
    se_mean = np.sqrt(MSE * (1/n + (x0 - x_mean)**2 / Sxx))
    se_pred = np.sqrt(MSE * (1 + 1/n + (x0 - x_mean)**2 / Sxx))
    ICM_Y = (y_pred - t_crit*se_mean, y_pred + t_crit*se_mean)
    IP_Y = (y_pred - t_crit*se_pred, y_pred + t_crit*se_pred)
    
    resultados.append([
        round(x0, 4),
        round(y_pred, 4),
        round(varianza, 4),
        round(R2, 4),
        round(r, 4),
        f"({round(IC_b1[0], 4)}, {round(IC_b1[1], 4)})",
        f"({round(IC_b0[0], 4)}, {round(IC_b0[1], 4)})",
        f"({round(ICM_Y[0], 4)}, {round(ICM_Y[1], 4)})",
        f"({round(IP_Y[0], 4)}, {round(IP_Y[1], 4)})"
    ])

columnas = [
    "X (perimetro_promedio)",
    "Ŷ = b0 + b1 x",
    "σ²",
    "R²",
    "r",
    "IC(β1)",
    "IC(β0)",
    "ICM(Y)",
    "IP(Y)"
]

tabla_resultados = pd.DataFrame(resultados, columns=columnas)
print(tabla_resultados)

# ===============================
# Mostrar tabla con Matplotlib
# ===============================
fig, ax = plt.subplots(figsize=(16, len(tabla_resultados)*0.8 + 1))
ax.axis('off')

tabla = ax.table(
    cellText=tabla_resultados.values,
    colLabels=tabla_resultados.columns,
    cellLoc='center',
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 1.5)  # más alto para mejor lectura
tabla.auto_set_column_width([i for i in range(len(columnas))])

plt.title(f"Análisis de regresión lineal — Variable X: {predictora}", fontsize=13, pad=20)
plt.show()
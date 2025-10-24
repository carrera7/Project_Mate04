# ============================================================
# REGRESIÓN LINEAL MÚLTIPLE - MÉTODO DEL DESCENSO DEL GRADIENTE
# ============================================================

import pandas as pd
import numpy as np
import json

# === 1. Cargar el dataset ===
df = pd.read_csv("./CSV/wdbc_full.csv")

# === 2. Definir variable dependiente (Y) e independientes (X) ===
Y = df["radio_promedio"].values
X = df[["area_promedio", "perimetro_promedio", "concavidad_promedio"]].values

n = len(Y)  # número de observaciones
m = X.shape[1]  # número de variables independientes

# === 3. Normalizar los datos (para mejor convergencia) ===
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean()) / Y.std()

# === 4. Inicializar parámetros ===
beta_0 = 0
beta = np.zeros(m)
alpha = 0.005      # tasa de aprendizaje
epochs = 10000     # número de iteraciones

# === 5. Definir funciones auxiliares ===
def calcular_costo(X, Y, beta_0, beta):
    Y_pred = beta_0 + np.dot(X, beta)
    error = Y_pred - Y
    J = (1 / (2 * len(Y))) * np.sum(error ** 2)
    return J

def calcular_R2(Y, Y_pred):
    sse = np.sum((Y - Y_pred) ** 2)
    sst = np.sum((Y - np.mean(Y)) ** 2)
    return 1 - sse / sst

# === 6. Descenso del gradiente ===
cost_history = []
for i in range(epochs):
    Y_pred = beta_0 + np.dot(X, beta)
    error = Y_pred - Y

    # Gradientes
    d_beta_0 = (1 / n) * np.sum(error)
    d_beta = (1 / n) * np.dot(X.T, error)

    # Actualización de parámetros
    beta_0 -= alpha * d_beta_0
    beta -= alpha * d_beta

    # Guardar costo cada 100 iteraciones
    if i % 100 == 0:
        cost = calcular_costo(X, Y, beta_0, beta)
        cost_history.append(cost)
        print(f"Iteración {i}: Costo = {cost:.6f}")

# === 7. Calcular métricas finales ===
Y_pred_final = beta_0 + np.dot(X, beta)
MSE = np.mean((Y - Y_pred_final) ** 2)
R2 = calcular_R2(Y, Y_pred_final)

# === 8. Guardar resultados ===
resultado_gradiente = {
    "metodo": "Regresión Lineal Múltiple - Descenso del Gradiente2",
    "coeficientes": {
        "beta_0": float(beta_0),
        "beta_1": float(beta[0]),
        "beta_2": float(beta[1]),
        "beta_3": float(beta[2])
    },
    "MSE": float(MSE),
    "R2": float(R2),
    "costo_final": float(cost_history[-1])
}

with open("resultado_gradiente.json", "w", encoding="utf-8") as f:
    json.dump(resultado_gradiente, f, indent=4, ensure_ascii=False)

print("\n✅ RESULTADOS FINALES (DESCENSO DEL GRADIENTE):")
print(json.dumps(resultado_gradiente, indent=4, ensure_ascii=False))
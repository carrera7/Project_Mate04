# ============================================================
# REGRESIÓN LINEAL MÚLTIPLE - MÉTODO DEL DESCENSO DEL GRADIENTE
# ============================================================

import pandas as pd
import numpy as np
import json

# === 1. Cargar el dataset ===
# Es necesario que primero se ejecute python new_csv.py
df = pd.read_csv("CSV/wdbc_selected.csv")

# === 2. Definir variable dependiente (Y) e independientes (X) ===
Y = df["radio_promedio"].values
X = df[["area_promedio", "perimetro_promedio", "concavidad_promedio"]].values

n = len(Y)  # número de observaciones

# === 3. Normalizar los datos ===
# Normalizar ayuda a que el algoritmo converja más rápido y estable
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean()) / Y.std()

# === 4. Inicializar los parámetros ===
beta_0 = 0
beta = np.zeros(X.shape[1])  # [β1, β2, β3]
alpha = 0.005  # tasa de aprendizaje
epochs = 10000  # número de iteraciones

# === 5. Definir la función de costo (Error Cuadrático Medio) ===
def calcular_costo(X, Y, beta_0, beta):
    Y_pred = beta_0 + np.dot(X, beta)
    error = Y_pred - Y
    J = (1 / (2 * len(Y))) * np.sum(error ** 2)
    return J

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

    # Guardar costo en cada iteración
    if i % 100 == 0:
        cost = calcular_costo(X, Y, beta_0, beta)
        cost_history.append(cost)
        print(f"Iteración {i}: Costo = {cost:.6f}")

# === 7. Resultados finales ===
resultado = {
    "metodo": "Regresión Lineal Múltiple - Descenso del Gradiente",
    "coeficientes": {
        "beta_0": float(beta_0),
        "beta_1": float(beta[0]),
        "beta_2": float(beta[1]),
        "beta_3": float(beta[2])
    },
    "costo_final": float(cost_history[-1])
}

# === 8. Guardar resultados en archivo JSON ===
with open("resultado_gradiente.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

print("\n✅ RESULTADOS FINALES:")
print(json.dumps(resultado, indent=4, ensure_ascii=False))

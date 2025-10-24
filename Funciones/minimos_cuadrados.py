# ============================================================
# REGRESIÓN LINEAL MÚLTIPLE - MÉTODO DE MÍNIMOS CUADRADOS
# ============================================================

import pandas as pd
import numpy as np
import json

# 1. Cargar el dataset 
df = pd.read_csv("./CSV/wdbc_full.csv")

# Definir variable dependiente (Y) e independientes (X) 
Y = df["radio_promedio"].values
X = df[["area_promedio", "perimetro_promedio", "concavidad_promedio"]].values

# Agregar columna de 1's para el intercepto
X_b = np.c_[np.ones((len(X), 1)), X]  # Agrega una columna de unos a la izquierda

# Calcular coeficientes por Mínimos Cuadrados
# Fórmula: β = (XᵀX)⁻¹ XᵀY
beta_opt = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y

# Extraer coeficientes
beta_0 = beta_opt[0]
beta_1, beta_2, beta_3 = beta_opt[1:]

# Calcular predicciones y métricas
Y_pred = X_b @ beta_opt
residuos = Y - Y_pred

# Error cuadrático medio (MSE)
MSE = np.mean(residuos ** 2)

# R² = 1 - (SSE/SST)
SSE = np.sum(residuos ** 2)
SST = np.sum((Y - np.mean(Y)) ** 2)
R2 = 1 - (SSE / SST)

# Guardar resultados en JSON 
resultado_mco = {
    "metodo": "Regresión Lineal Múltiple - Mínimos Cuadrados Ordinarios ",
    "coeficientes": {
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_2": float(beta_2),
        "beta_3": float(beta_3)
    },
    "MSE": float(MSE),
    "R2": float(R2)
}

with open("resultado_mco.json", "w", encoding="utf-8") as f:
    json.dump(resultado_mco, f, indent=4, ensure_ascii=False)

print("\n RESULTADOS FINALES (MCO):")
print(json.dumps(resultado_mco, indent=4, ensure_ascii=False))
# ============================================================
# REGRESIÓN LINEAL MÚLTIPLE NORMALIZADA - MÉTODO DE MÍNIMOS CUADRADOS
# ============================================================

import pandas as pd
import numpy as np
import json

# === 1. Cargar el dataset ===
df = pd.read_csv("./CSV/wdbc_full.csv")

# === 2. Definir variable dependiente (Y) e independientes (X) ===
Y = df["radio_promedio"].values
X = df[["area_promedio", "perimetro_promedio", "concavidad_promedio"]].values

# === 3. Normalizar los datos ===
# Normalización tipo Z-score: (x - media) / desviación estándar
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
Y_mean = Y.mean()
Y_std = Y.std()

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# === 4. Agregar columna de 1's para el intercepto ===
X_b = np.c_[np.ones((len(X_norm), 1)), X_norm]

# === 5. Calcular coeficientes por Mínimos Cuadrados ===
# Fórmula: β = (XᵀX)⁻¹ XᵀY
beta_opt = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_norm

# === 6. Extraer coeficientes ===
beta_0 = beta_opt[0]
beta_1, beta_2, beta_3 = beta_opt[1:]

# === 7. Calcular predicciones y métricas ===
Y_pred_norm = X_b @ beta_opt

# Volver a la escala original (opcional)
Y_pred = (Y_pred_norm * Y_std) + Y_mean

# Cálculo de residuos en escala original
residuos = Y - Y_pred

# Error cuadrático medio (MSE)
MSE = np.mean(residuos ** 2)

# R² = 1 - (SSE/SST)
SSE = np.sum(residuos ** 2)
SST = np.sum((Y - np.mean(Y)) ** 2)
R2 = 1 - (SSE / SST)

# === 8. Guardar resultados en JSON ===
resultado_mco_norm = {
    "metodo": "Regresión Lineal Múltiple - Mínimos Cuadrados Ordinarios (Normalizado)",
    "coeficientes_normalizados": {
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_2": float(beta_2),
        "beta_3": float(beta_3)
    },
    "MSE": float(MSE),
    "R2": float(R2),
    "media_X": X_mean.tolist(),
    "desviacion_X": X_std.tolist(),
    "media_Y": float(Y_mean),
    "desviacion_Y": float(Y_std)
}

with open("resultado_mco_normalizado.json", "w", encoding="utf-8") as f:
    json.dump(resultado_mco_norm, f, indent=4, ensure_ascii=False)

print("\n✅ RESULTADOS FINALES (MCO NORMALIZADO):")
print(json.dumps(resultado_mco_norm, indent=4, ensure_ascii=False))
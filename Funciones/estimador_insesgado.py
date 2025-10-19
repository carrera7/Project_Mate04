import pandas as pd
import json
import numpy as np
import os

# === Rutas de archivos ===
ruta_json = "resultado.json"
ruta_csv_tabla = "tabla_regresiones_completa.csv"
ruta_csv_datos = "CSV/wdbc_selected.csv"  # Para calcular r (corr(x,y)) con los datos originales

# === 1. Cargar JSON y CSV existentes ===
if not os.path.exists(ruta_json):
    raise FileNotFoundError("❌ No se encontró 'resultado.json'.")

if not os.path.exists(ruta_csv_tabla):
    raise FileNotFoundError("❌ No se encontró 'tabla_regresiones_completa.csv'.")

with open(ruta_json, "r", encoding="utf-8") as f:
    resultado = json.load(f)

df_tabla = pd.read_csv(ruta_csv_tabla)
df_datos = pd.read_csv(ruta_csv_datos)

# === 2. Definir parámetros desde el JSON y estructura de la tabla ===
predictoras = ["radio_promedio", "perimetro_promedio", "concavidad_promedio"]
y_col = "area_promedio"
n = int(resultado["n_observaciones"])  # 569

# Validación del tamaño esperado del CSV (3 bloques de n filas)
filas_totales = len(df_tabla)
esperadas = n * len(predictoras)
if filas_totales != esperadas:
    raise ValueError(f"❌ La tabla tiene {filas_totales} filas, pero se esperaban {esperadas} (= {n} × {len(predictoras)}). Verifica el CSV o el orden de concatenación.")

# === 3. Procesar predictor por predictor en bloques de n filas ===
for k, x_col in enumerate(predictoras):
    print(f"\n📘 Procesando predictor: {x_col}")

    # --- Selección de bloque (sin recalcular ŷ) ---
    # Bloque k: filas [start : end) del CSV
    start = k * n
    end = (k + 1) * n
    bloque = df_tabla.iloc[start:end]

    # Variables desde el CSV existente:
    # - Y (observado) y ŷ (estimado) ya están en la tabla
    y_real = bloque["Y"].values
    y_est = bloque["y_i = β1x_i + β0"].values

    # === PASO A PASO DE FÓRMULAS ===

    # (1) Residuos por observación
    # e_i = y_i - ŷ_i
    e_i = y_real - y_est

    # (2) Suma de cuadrados de los errores
    # SS_e = Σ e_i^2
    SS_e = float(np.sum(e_i ** 2))

    # (3) Suma de cuadrados corregida respecto a la media de Y
    # SCE = Σ (y_i - \bar{y})^2 , con \bar{y} media del bloque
    y_media_bloque = float(np.mean(y_real))
    SCE = float(np.sum((y_real - y_media_bloque) ** 2))

    # (4) Estimador insesgado de la varianza del error
    # S^2 = SS_e / (n - 2)
    S2 = SS_e / (n - 2)

    # (5) Varianza relativa alternativa (según apuntes)
    # σ̂^2_rel = SS_e / SCE
    sigma2_rel = SS_e / SCE

    # (6) Coeficiente de determinación
    # R^2 = 1 - (SS_e / SCE)
    R2 = 1 - (SS_e / SCE)

    # (7) Coeficiente de correlación (Pearson) entre X e Y
    # r = corr(x, y) usando datos originales del CSV de datos
    r = float(np.corrcoef(df_datos[x_col], df_datos[y_col])[0, 1])

    # === 4. Guardar resultados en el JSON bajo el predictor ===
    # Nota: mantenemos y complementamos la estructura que ya tienes
    if "coeficientes" not in resultado or x_col not in resultado["coeficientes"]:
        raise KeyError(f"❌ No se encontraron coeficientes para '{x_col}' en el JSON.")

    resultado["coeficientes"][x_col]["SS_e"] = SS_e
    resultado["coeficientes"][x_col]["SCE"] = SCE
    resultado["coeficientes"][x_col]["S2_insesgado"] = S2          # S^2 (estimador insesgado)
    resultado["coeficientes"][x_col]["sigma2_relativo"] = sigma2_rel
    resultado["coeficientes"][x_col]["R2"] = R2
    resultado["coeficientes"][x_col]["r"] = r
    # Mantener la fórmula de regresión ya presente en el JSON (no se modifica)

    # === 5. Actualizar el CSV existente por bloque ===
    # Importante: la cabecera mostrada indica 'σ²', no 'σ̂'.
    # Rellenamos columnas del modelo por bloque (constantes por predictor).
    df_tabla.loc[start:end-1, "σ²"] = S2       # varianza insesgada del error
    df_tabla.loc[start:end-1, "R²"] = R2
    df_tabla.loc[start:end-1, "r"] = r
    # No se agregan nuevas columnas; se mantiene la estructura original.

# === 6. Guardar JSON y CSV actualizados ===
with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

df_tabla.to_csv(ruta_csv_tabla, index=False, encoding="utf-8")

print("\n✅ JSON actualizado con SS_e, SCE, S² (σ² insesgada), σ² relativa, R² y r por predictor")
print("✅ CSV actualizado: columnas 'σ²', 'R²' y 'r' completadas por bloque de predictor")
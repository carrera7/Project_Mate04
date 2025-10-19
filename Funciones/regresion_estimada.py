import pandas as pd
import json
import os
import numpy as np

# === 1. Cargar el archivo JSON con coeficientes ===
ruta_json = "resultado.json"

if not os.path.exists(ruta_json):
    raise FileNotFoundError("❌ No se encontró 'resultado.json'. Ejecuta primero los scripts de medias y coeficientes.")

with open(ruta_json, "r", encoding="utf-8") as f:
    resultado = json.load(f)

# === 2. Cargar el CSV con los datos ===
ruta_csv = "CSV/wdbc_selected.csv"
df = pd.read_csv(ruta_csv)

# === 3. Definir las variables ===
predictoras = ["radio_promedio", "perimetro_promedio", "concavidad_promedio"]
y_col = "area_promedio"

# === 4. Crear tabla con la misma estructura que la foto ===
columnas = ["Y", "y_i = β1x_i + β0", "σ̂", "R²", "r", "IC(β1)", "IC(β0)", "ICM(Y)", "IP(Y)"]
tabla_final = pd.DataFrame(columns=columnas)

for x_col in predictoras:
    print(f"\n📘 Calculando regresión estimada para: {x_col}")

    # Obtener coeficientes desde el JSON
    beta_0 = resultado["coeficientes"][x_col]["beta_0"]
    beta_1 = resultado["coeficientes"][x_col]["beta_1"]

    # Calcular valores estimados
    y_estimada = beta_0 + beta_1 * df[x_col]

    # Guardar fórmula en el JSON
    formula = f"Y = {beta_0:.4f} + {beta_1:.4f} * X"
    resultado["coeficientes"][x_col]["regresion_estimada"] = formula

    # Crear filas para esta variable predictora
    temp = pd.DataFrame({
        "Y": df[y_col],
        "y_i = β1x_i + β0": y_estimada,
        "σ̂": np.nan,
        "R²": np.nan,
        "r": np.nan,
        "IC(β1)": np.nan,
        "IC(β0)": np.nan,
        "ICM(Y)": np.nan,
        "IP(Y)": np.nan
    })

    # Agregar al DataFrame final
    tabla_final = pd.concat([tabla_final, temp], ignore_index=True)

# === 5. Guardar el JSON actualizado ===
with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

# === 6. Exportar la tabla final al CSV ===
ruta_salida_csv = "tabla_regresiones_completa.csv"
tabla_final.to_csv(ruta_salida_csv, index=False, encoding="utf-8")

print("\n✅ Tabla con estructura igual a la foto exportada en:", ruta_salida_csv)
print(tabla_final.head())
import pandas as pd
import os
import json

# === 1. Cargar el dataset ===
# Ruta a tu archivo CSV (ajustada a tu estructura de carpetas)
df = pd.read_csv("CSV/wdbc_selected.csv")

# === 2. Seleccionar las variables predictoras ===
columnas = ["radio_promedio", "perimetro_promedio", "concavidad_promedio"]

# === 3. Calcular manualmente el número total de observaciones ===
n = len(df)

# === 4. Calcular las medias paso a paso ===
# Fórmula teórica:  x̄ = (Σxi) / n

medias = {}
for col in columnas:
    suma = df[col].sum()               # Σxi
    media = suma / n                   # x̄ = Σxi / n
    medias[col] = float(media)

    # Mostrar cada cálculo con detalle
    print(f"\n📘 Variable: {col}")
    print(f"  → Suma total (Σxi): {suma:.4f}")
    print(f"  → Número de observaciones (n): {n}")
    print(f"  → Media calculada (x̄): {media:.4f}")

# === 5. Crear o cargar el archivo JSON existente ===
ruta_json = "resultado.json"

if os.path.exists(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        resultado = json.load(f)
else:
    resultado = {}

# === 6. Guardar los resultados ===
resultado["medias"] = medias
resultado["n_observaciones"] = n

# === 7. Escribir los datos en JSON (actualizando o creando el archivo) ===
with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

# === 8. Confirmar resultados ===
print("\n✅ Resultados guardados en 'resultado.json'")
print(json.dumps(resultado, indent=4, ensure_ascii=False))

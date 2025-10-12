import pandas as pd
import os
import json

# === 1. Cargar el dataset ===
# Asegurate de ajustar la ruta correctamente
df = pd.read_csv("CSV/wdbc_selected.csv")

# === 2. Definir columnas predictoras y la variable respuesta ===
columnas_predictoras = ["radio_promedio", "perimetro_promedio", "concavidad_promedio"]
columna_respuesta = "area_promedio"

# === 3. Calcular manualmente el número total de observaciones ===
n = len(df)

# === 4. Calcular las medias paso a paso ===
# Fórmula teórica: x̄ = (Σxi) / n
medias = {}

# Calcular medias de las variables predictoras
for col in columnas_predictoras:
    suma = df[col].sum()               # Σxi
    media = suma / n                   # x̄ = Σxi / n
    medias[col] = float(media)

    # Mostrar cada cálculo con detalle
    print(f"\n📘 Variable predictora: {col}")
    print(f"  → Suma total (Σxi): {suma:.4f}")
    print(f"  → Número de observaciones (n): {n}")
    print(f"  → Media calculada (x̄): {media:.4f}")

# Calcular media de la variable respuesta (Y)
suma_y = df[columna_respuesta].sum()
media_y = suma_y / n
medias[columna_respuesta] = float(media_y)

print(f"\n📗 Variable respuesta: {columna_respuesta}")
print(f"  → Suma total (Σyi): {suma_y:.4f}")
print(f"  → Número de observaciones (n): {n}")
print(f"  → Media calculada (ȳ): {media_y:.4f}")

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

# === 7. Escribir los datos en JSON ===
with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

# === 8. Confirmar resultados ===
print("\n✅ Resultados guardados en 'resultado.json'")
print(json.dumps(resultado, indent=4, ensure_ascii=False))

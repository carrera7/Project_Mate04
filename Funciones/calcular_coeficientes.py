import pandas as pd
import json
import os

# === 1. Cargar archivo JSON con medias y n ===
ruta_json = "resultado.json"

if not os.path.exists(ruta_json):
    raise FileNotFoundError("❌ No se encontró 'resultado.json'. Ejecuta primero el script de calcular_medias.py")

with open(ruta_json, "r", encoding="utf-8") as f:
    resultado = json.load(f)

# === 2. Cargar el dataset ===
df = pd.read_csv("CSV/wdbc_selected.csv")

# === 3. Definir columnas predictoras y respuesta ===
predictoras = ["radio_promedio", "perimetro_promedio", "concavidad_promedio"]
y_col = "area_promedio"

# === 4. media de Y ===
media_y = resultado["medias"]["area_promedio"]

# === 5. Crear diccionario para almacenar los coeficientes ===
coeficientes = {}

# === 6. Calcular Sxx, Sxy, β1 y β0 para cada predictor ===
for x_col in predictoras:
    media_x = resultado["medias"][x_col]
    
    Sxx = 0
    Sxy = 0
    
    # Cálculo manual
    for i in range(len(df)):
        xi = df.loc[i, x_col]
        yi = df.loc[i, y_col]
        Sxx += (xi - media_x) ** 2
        Sxy += (xi - media_x) * (yi - media_y)
    
    beta_1 = Sxy / Sxx
    beta_0 = media_y - beta_1 * media_x

    coeficientes[x_col] = {
        "Sxx": float(Sxx),
        "Sxy": float(Sxy),
        "beta_1": float(beta_1),
        "beta_0": float(beta_0),
        "media_y": float(media_y)
    }

    print(f"\n📘 Variable: {x_col}")
    print(f"  Sxx = {Sxx:.4f}")
    print(f"  Sxy = {Sxy:.4f}")
    print(f"  β₁ (pendiente) = {beta_1:.6f}")
    print(f"  β₀ (intercepto) = {beta_0:.6f}")

# === 7. Guardar en el JSON ===
resultado["coeficientes"] = coeficientes

with open(ruta_json, "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

# === 8. Confirmar ===
print("\n✅ Coeficientes calculados para todas las variables y guardados en 'resultado.json'")

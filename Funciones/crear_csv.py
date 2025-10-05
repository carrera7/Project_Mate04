# ============================================
# Crear un CSV completo con encabezados en español
# Dataset: Breast Cancer Wisconsin Diagnostic
# Autor: Josue
# ============================================

import pandas as pd

# === Nombres de las columnas traducidos ===
columnas = [
    "id", "diagnostico",  # ID del paciente y tipo de tumor (M = maligno, B = benigno)

    # Promedios (mean)
    "radio_promedio", "textura_promedio", "perimetro_promedio", "area_promedio", "suavidad_promedio",
    "compacidad_promedio", "concavidad_promedio", "puntos_concavos_promedio", "simetria_promedio", "dimension_fractal_promedio",

    # Error estándar (se)
    "radio_error", "textura_error", "perimetro_error", "area_error", "suavidad_error",
    "compacidad_error", "concavidad_error", "puntos_concavos_error", "simetria_error", "dimension_fractal_error",

    # Valores peores (worst)
    "radio_peor", "textura_peor", "perimetro_peor", "area_peor", "suavidad_peor",
    "compacidad_peor", "concavidad_peor", "puntos_concavos_peor", "simetria_peor", "dimension_fractal_peor"
]

# === Leer el archivo original sin encabezados ===
# Ruta relativa desde la carpeta 'Funciones' hacia 'Data'
ruta_datos = "Data/wdbc.data"
df = pd.read_csv(ruta_datos, header=None, names=columnas)

# === Guardar el nuevo archivo CSV con encabezado en español ===
ruta_salida = "CSV/wdbc_full.csv"
df.to_csv(ruta_salida, index=False, encoding="utf-8")

# === Mostrar resumen en pantalla ===
print("✅ Archivo CSV creado correctamente.")
print(f"Ruta: {ruta_salida}")
print(f"Filas: {len(df)}")
print(f"Columnas: {len(df.columns)}")
print("\nPrimeras filas del dataset:\n")
print(df.head())

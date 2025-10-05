import pandas as pd

# Cargar el CSV original
df = pd.read_csv("CSV/wdbc_full.csv")

# Columnas de interés (respuesta y predictoras)
columnas = ['area_promedio', 'radio_promedio', 'perimetro_promedio', 'concavidad_promedio']

# Seleccionar solo esas columnas
nuevo_df = df[columnas]

# Guardar el nuevo CSV
nuevo_df.to_csv("CSV/wdbc_selected.csv", index=False)

print("Nuevo CSV creado correctamente. Ruta: CSV/wdbc_selected.csv")
print(nuevo_df.head())

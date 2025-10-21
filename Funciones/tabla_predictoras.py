import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# ===============================
# Configuración
# ===============================
archivo_csv = "CSV/wdbc_full.csv"
respuesta = "radio_promedio"
predictoras = ["perimetro_promedio", "area_promedio", "radio_peor", "perimetro_peor"]
alpha = 0.05  # Nivel de significancia para IC (95%)

# ===============================
# Cargar datos
# ===============================
df = pd.read_csv(archivo_csv)

# ===============================
# Función regresión lineal simple
# ===============================
def simple_linear_regression(x, y, alpha=0.05):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    Sxx = np.sum((x - x_mean) ** 2)
    Syy = np.sum((y - y_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))

    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean

    y_pred = b0 + b1 * x
    residuals = y - y_pred

    SSR = np.sum(residuals ** 2)  # suma de residuos
    sigma2 = SSR / (n - 2)
    sigma = np.sqrt(sigma2)

    R2 = 1 - SSR / Syy
    r = np.sqrt(R2) * np.sign(b1)

    # Error estándar
    se_b1 = np.sqrt(sigma2 / Sxx)
    se_b0 = np.sqrt(sigma2 * (1/n + x_mean**2 / Sxx))

    t_crit = t.ppf(1 - alpha/2, n - 2)

    # Intervalos de confianza
    IC_b1 = (b1 - t_crit * se_b1, b1 + t_crit * se_b1)
    IC_b0 = (b0 - t_crit * se_b0, b0 + t_crit * se_b0)

    # Intervalo de confianza y predicción para Y en x̄
    x_star = x_mean
    y_hat_star = b0 + b1 * x_star
    se_Yhat = np.sqrt(sigma2 * (1/n + (x_star - x_mean)**2 / Sxx))
    se_pred = np.sqrt(sigma2 * (1 + 1/n + (x_star - x_mean)**2 / Sxx))

    ICY = (y_hat_star - t_crit * se_Yhat, y_hat_star + t_crit * se_Yhat)
    IPY = (y_hat_star - t_crit * se_pred, y_hat_star + t_crit * se_pred)

    return {
        "b0": b0,
        "b1": b1,
        "sigma2": sigma2,
        "R2": R2,
        "r": r,
        "IC_b1": IC_b1,
        "IC_b0": IC_b0,
        "ICM_Y": ICY,
        "IP_Y": IPY
    }

# ===============================
# Calcular tabla
# ===============================
rows = []
for col in predictoras:
    data = df[[col, respuesta]].dropna()
    x = data[col].values.astype(float)
    y = data[respuesta].values.astype(float)
    res = simple_linear_regression(x, y, alpha)
    rows.append({
        "Variable X": col,
        "ŷ = b0 + b1 x": f"{res['b0']:.4f} + {res['b1']:.4f} x",
        "σ²": f"{res['sigma2']:.4f}",
        "R²": f"{res['R2']:.4f}",
        "r": f"{res['r']:.4f}",
        "IC(β1)": f"({res['IC_b1'][0]:.4f}, {res['IC_b1'][1]:.4f})",
        "IC(β0)": f"({res['IC_b0'][0]:.4f}, {res['IC_b0'][1]:.4f})",
        "ICM(Y)": f"({res['ICM_Y'][0]:.4f}, {res['ICM_Y'][1]:.4f})",
        "IP(Y)": f"({res['IP_Y'][0]:.4f}, {res['IP_Y'][1]:.4f})"
    })

tabla = pd.DataFrame(rows)
print(tabla)

# ===============================
# Graficar tabla con Matplotlib
# ===============================
fig, ax = plt.subplots(figsize=(14, len(tabla) * 0.8 + 1))
ax.axis('tight')
ax.axis('off')

# Crear la tabla
the_table = ax.table(
    cellText=tabla.values,
    colLabels=tabla.columns,
    loc='center',
    cellLoc='center'
)

# Ajustar el tamaño de la fuente y la escala de la tabla
the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
the_table.scale(1, 1.5)

# Ajustar el ancho de las primeras dos columnas
# El primer valor corresponde a la primera columna, el segundo a la segunda, y el resto son valores estándar
the_table.auto_set_column_width([0, 1] + list(range(2, len(tabla.columns))))

# Título
plt.title(f"Análisis de regresión lineal simple (Y = {respuesta})", fontsize=12, pad=20)
plt.show()
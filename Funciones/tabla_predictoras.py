import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# ===============================
# Configuración
# ===============================
archivo_csv = "CSV/wdbc_full.csv"  # Ruta del archivo CSV con los datos
respuesta = "radio_promedio"  # Variable dependiente
predictoras = ["perimetro_promedio", "area_promedio", "radio_peor", "perimetro_peor"]  # Lista de variables predictoras

df = pd.read_csv(archivo_csv)

# ===============================
# Preparar tabla resumen
# ===============================
resultados = []

# Bucle sobre cada variable predictora
for predictora in predictoras:
    data = df[[predictora, respuesta]].dropna()
    x = data[predictora].values
    y = data[respuesta].values
    n = len(x)

    # Cálculos de medias
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Sumas de cuadrados y productos cruzados
    Sxx = np.sum((x - x_mean)**2)
    Sxy = np.sum((x - x_mean)*(y - y_mean))
    Syy = np.sum((y - y_mean)**2)

    # Estimación de parámetros
    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean
    y_pred = b0 + b1 * x
    resid = y - y_pred

    # SSE, R² y r
    SSE = np.sum(resid**2)
    R2 = 1 - SSE / Syy
    r = np.sign(b1) * np.sqrt(R2)

    # Varianza de errores
    sigma2 = SSE / (n - 2)

    # Valor crítico t (95%)
    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, n - 2)

    # IC para beta1
    se_b1 = np.sqrt(sigma2 / Sxx)
    IC_b1 = (b1 - t_crit*se_b1, b1 + t_crit*se_b1)

    # IC para beta0
    se_b0 = np.sqrt(sigma2 * (1/n + x_mean**2/Sxx))
    IC_b0 = (b0 - t_crit*se_b0, b0 + t_crit*se_b0)

    # ===============================
    # Intervalos para Y en x = x_mean
    # ===============================
    y_hat_mean = b0 + b1 * x_mean
    se_yhat = np.sqrt(sigma2 * (1/n))  # en x = x_mean, (x - x_mean)^2 = 0

    ICM_y = (y_hat_mean - t_crit*se_yhat, y_hat_mean + t_crit*se_yhat)
    IP_y = (y_hat_mean - t_crit*np.sqrt(sigma2 + se_yhat**2),
            y_hat_mean + t_crit*np.sqrt(sigma2 + se_yhat**2))

    # Guardar resultados
    resultados.append({
        "Predictora": predictora,
        "β1": b1,
        "β0": b0,
        "σ²": sigma2,
        "R²": R2,
        "r": r,
        "IC(β1)": f"[{IC_b1[0]:.3f}, {IC_b1[1]:.3f}]",
        "IC(β0)": f"[{IC_b0[0]:.3f}, {IC_b0[1]:.3f}]",
        "ICM(Y)": f"[{ICM_y[0]:.3f}, {ICM_y[1]:.3f}]",
        "IP(Y)": f"[{IP_y[0]:.3f}, {IP_y[1]:.3f}]"
    })

# ===============================
# Crear DataFrame resumen final
# ===============================
tabla = pd.DataFrame(resultados)
tabla = tabla.round({"β1":3,"β0":3,"σ²":3,"R²":3,"r":3})
print(tabla)

# ===============================
# Dibujar tabla
# ===============================
fig, ax = plt.subplots(figsize=(14, 2 + len(tabla)*0.6))  # tamaño dinámico
ax.axis('off')

tabla_grafica = ax.table(
    cellText=tabla.values,
    colLabels=tabla.columns,
    cellLoc='center',
    loc='center'
)

tabla_grafica.auto_set_font_size(False)
tabla_grafica.set_fontsize(10)
tabla_grafica.scale(1, 1.4)

plt.tight_layout()
plt.show()

fig.savefig("tabla_regresion_multiple.png", dpi=400)
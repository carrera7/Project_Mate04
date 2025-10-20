import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# ===============================
# Datos de ejemplo
# ===============================
archivo_csv = "CSV/wdbc_full.csv"
respuesta = "radio_promedio"
predictora = "perimetro_promedio"

df = pd.read_csv(archivo_csv)
data = df[[predictora, respuesta]].dropna()
x = data[predictora].values
y = data[respuesta].values
n = len(x)

x_mean = np.mean(x)
y_mean = np.mean(y)

Sxx = np.sum((x - x_mean)**2)
Sxy = np.sum((x - x_mean)*(y - y_mean))
Syy = np.sum((y - y_mean)**2)

b1 = Sxy / Sxx
b0 = y_mean - b1 * x_mean
y_pred = b0 + b1 * x
resid = y - y_pred

SSE = np.sum(resid**2)
R2 = 1 - SSE/Syy
r = np.sign(b1) * np.sqrt(R2)

sigma2 = SSE / (n - 2)
sigma = np.sqrt(sigma2)

# Valor crítico t (95%)
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, n - 2)

# Intervalo para beta1
se_b1 = np.sqrt(sigma2 / Sxx)
IC_b1 = (b1 - t_crit*se_b1, b1 + t_crit*se_b1)

# Intervalo para beta0
se_b0 = np.sqrt(sigma2 * (1/n + x_mean**2/Sxx))
IC_b0 = (b0 - t_crit*se_b0, b0 + t_crit*se_b0)

# Intervalos para cada punto
ICM = []
IP = []
for xi, yhat in zip(x, y_pred):
    se_mean = np.sqrt(sigma2 * (1/n + (xi - x_mean)**2 / Sxx))
    se_pred = np.sqrt(sigma2 * (1 + 1/n + (xi - x_mean)**2 / Sxx))
    ICM.append((yhat - t_crit*se_mean, yhat + t_crit*se_mean))
    IP.append((yhat - t_crit*se_pred, yhat + t_crit*se_pred))

# ===============================
# Crear DataFrame resumen
# ===============================
tabla = pd.DataFrame({
    "Y": y,
    "ŷ": y_pred,
    "β1": [b1]*n,
    "β0": [b0]*n,
    "σ²": [sigma2]*n,
    "R²": [R2]*n,
    "r": [r]*n,
    "IC(β1)": [f"[{IC_b1[0]:.3f}, {IC_b1[1]:.3f}]"]*n,
    "IC(β0)": [f"[{IC_b0[0]:.3f}, {IC_b0[1]:.3f}]"]*n,
    "ICM(Y)": [f"[{a:.3f}, {b:.3f}]" for a,b in ICM],
    "IP(Y)": [f"[{a:.3f}, {b:.3f}]" for a,b in IP]
})

# Redondear para mejor visualización
tabla = tabla.round({"Y":3,"ŷ":3,"β1":3,"β0":3,"σ²":3,"R²":3,"r":3})
print(tabla.head())

# dibujar tabla
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

tabla_grafica = ax.table(
    cellText=tabla.head(10).values,  # primeras filas (puedes usar todas si querés)
    colLabels=tabla.columns,
    cellLoc='center',
    loc='center'
)

tabla_grafica.auto_set_font_size(False)
tabla_grafica.set_fontsize(9)
tabla_grafica.scale(1, 1.5)  # agrandar filas

plt.tight_layout()
plt.show()

fig.savefig("tabla_regresion.png", dpi=300)



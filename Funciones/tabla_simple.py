import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# ===============================
# Datos de ejemplo
# ===============================
archivo_csv = "CSV/wdbc_full.csv"  # Ruta del archivo CSV con los datos
respuesta = "radio_promedio"  # Variable dependiente
predictora = "perimetro_promedio"  # Variable independiente

df = pd.read_csv(archivo_csv)  # Leer el archivo CSV
data = df[[predictora, respuesta]].dropna()  # Eliminar valores nulos
x = data[predictora].values  # Valores de la variable independiente
y = data[respuesta].values  # Valores de la variable dependiente
n = len(x)  # Número de datos

# ===============================
# Cálculos de la regresión
# ===============================

# Cálculos de medias
x_mean = np.mean(x)  # Media de X: \(\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i\)
y_mean = np.mean(y)  # Media de Y: \(\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i\)

# Sumas de cuadrados y productos cruzados
Sxx = np.sum((x - x_mean)**2)  # Suma de cuadrados de X: \(S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2\)
Sxy = np.sum((x - x_mean)*(y - y_mean))  # Suma de productos cruzados: \(S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})\)
Syy = np.sum((y - y_mean)**2)  # Suma de cuadrados de Y: \(S_{yy} = \sum_{i=1}^{n} (y_i - \bar{y})^2\)

# Estimación de los parámetros de la regresión
b1 = Sxy / Sxx  # Pendiente: \(\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}}\)
b0 = y_mean - b1 * x_mean  # Intercepto: \(\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \cdot \bar{x}\)

# Predicciones de Y: \(\hat{y_i} = \hat{\beta}_0 + \hat{\beta}_1 x_i\)
y_pred = b0 + b1 * x

# Cálculo de los residuos: \(e_i = y_i - \hat{y_i}\)
resid = y - y_pred

# Cálculo de la suma de los errores cuadráticos (SSE): \(SSE = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y_i})^2\)
SSE = np.sum(resid**2)

# Cálculo de R²: \(R^2 = 1 - \frac{SSE}{S_{yy}}\)
R2 = 1 - SSE / Syy

# Cálculo del coeficiente de correlación: \(r = \text{sign}(\hat{\beta}_1) \cdot \sqrt{R^2}\)
r = np.sign(b1) * np.sqrt(R2)

# Cálculo de la varianza de los errores: \(\sigma^2 = \frac{SSE}{n - 2}\)
sigma2 = SSE / (n - 2)

# Desviación estándar de los errores: \(\sigma = \sqrt{\sigma^2}\)
sigma = np.sqrt(sigma2)

# ===============================
# Intervalos de confianza
# ===============================

# Valor crítico t (95%): \(t_{\alpha/2, n-2}\) es el valor de la distribución t con \(n-2\) grados de libertad y \(\alpha = 0.05\)
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, n - 2)  # Valor crítico t para un intervalo de confianza del 95%

# Intervalo de confianza para β1: 
# \(IC(\hat{\beta}_1) = \left[ \hat{\beta}_1 - t_{\alpha/2, n-2} \cdot \text{SE}(\hat{\beta}_1), \hat{\beta}_1 + t_{\alpha/2, n-2} \cdot \text{SE}(\hat{\beta}_1) \right]\)
se_b1 = np.sqrt(sigma2 / Sxx)  # Error estándar de β1: \(\text{SE}(\hat{\beta}_1) = \sqrt{\frac{\sigma^2}{S_{xx}}}\)
IC_b1 = (b1 - t_crit*se_b1, b1 + t_crit*se_b1)  # Intervalo de confianza para β1

# Intervalo de confianza para β0: 
# \(IC(\hat{\beta}_0) = \left[ \hat{\beta}_0 - t_{\alpha/2, n-2} \cdot \text{SE}(\hat{\beta}_0), \hat{\beta}_0 + t_{\alpha/2, n-2} \cdot \text{SE}(\hat{\beta}_0) \right]\)
se_b0 = np.sqrt(sigma2 * (1/n + x_mean**2/Sxx))  # Error estándar de β0: \(\text{SE}(\hat{\beta}_0) = \sqrt{\frac{\sigma^2}{n} + \frac{\bar{x}^2}{S_{xx}}}\)
IC_b0 = (b0 - t_crit*se_b0, b0 + t_crit*se_b0)  # Intervalo de confianza para β0

# ===============================
# Intervalos para cada punto
# ===============================

# Intervalo de confianza para la media de Y:
# \(IC_{media}(y_i) = \left[ \hat{y_i} - t_{\alpha/2, n-2} \cdot SE_{media}(y_i), \hat{y_i} + t_{\alpha/2, n-2} \cdot SE_{media}(y_i) \right]\)
# Error estándar de la media: \(SE_{media}(y_i) = \sqrt{\sigma^2 \left( \frac{1}{n} + \frac{(x_i - \bar{x})^2}{S_{xx}} \right)}\)

# Intervalo de predicción para un nuevo valor de Y:
# \(IC_{pred}(y_i) = \left[ \hat{y_i} - t_{\alpha/2, n-2} \cdot SE_{pred}(y_i), \hat{y_i} + t_{\alpha/2, n-2} \cdot SE_{pred}(y_i) \right]\)
# Error estándar de la predicción: \(SE_{pred}(y_i) = \sqrt{\sigma^2 \left( 1 + \frac{1}{n} + \frac{(x_i - \bar{x})^2}{S_{xx}} \right)}\)

ICM = []
IP = []
for xi, yhat in zip(x, y_pred):
    se_mean = np.sqrt(sigma2 * (1/n + (xi - x_mean)**2 / Sxx))  # Error estándar de la media
    se_pred = np.sqrt(sigma2 * (1 + 1/n + (xi - x_mean)**2 / Sxx))  # Error estándar de la predicción
    ICM.append((yhat - t_crit*se_mean, yhat + t_crit*se_mean))  # Intervalo de confianza para la media de Y
    IP.append((yhat - t_crit*se_pred, yhat + t_crit*se_pred))  # Intervalo de predicción

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

# ===============================
# Dibujar tabla
# ===============================
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

fig.savefig("tabla_regresion.png", dpi=400)
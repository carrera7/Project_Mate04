import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



archivo_csv = "CSV/wdbc_full.csv"
respuesta = "radio_promedio"  # variable respuesta

df = pd.read_csv(archivo_csv)

predictoras = ["area_promedio", "perimetro_promedio", "radio_peor", "perimetro_peor"]


def simple_linear_regression(x, y):
    """
    Calcula regresión lineal simple y devuelve:
    intercepto a, pendiente b, y_pred, residuals, R2, SSE, SST, RMSE, se_b (error std de pendiente)
    """
    n = len(x)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 observaciones.")

    # medias
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # sumas necesarias
    Sxx = np.sum((x - x_mean) ** 2)                     # Σ(x - x̄)^2
    Sxy = np.sum((x - x_mean) * (y - y_mean))          # Σ(x - x̄)(y - ȳ)
    Syy = np.sum((y - y_mean) ** 2)                    # Σ(y - ȳ)^2

    # coeficientes
    b = Sxy / Sxx                                      # pendiente
    a = y_mean - b * x_mean                            # intercepto

    # predicciones y residuales
    y_pred = a + b * x
    residuals = y - y_pred

    # SSE, SST, SSR
    SSE = np.sum(residuals ** 2)                       # suma de errores al cuadrado
    SST = Syy                                          # suma total de cuadrados
    SSR = SST - SSE                                    # suma regresion

    # R^2
    R2 = 1 - (SSE / SST) if SST != 0 else np.nan

    # RMSE
    RMSE = np.sqrt(SSE / (n - 2))                      # error estándar del ajuste

    # Error estándar de la pendiente b: se_b = sqrt( s^2 / Sxx ), donde s^2 = SSE/(n-2)
    se_b = np.sqrt((SSE / (n - 2)) / Sxx) if (n > 2 and Sxx != 0) else np.nan

    return {
        "a": a, "b": b,
        "y_pred": y_pred,
        "residuals": residuals,
        "R2": R2,
        "SSE": SSE, "SST": SST, "SSR": SSR,
        "RMSE": RMSE,
        "se_b": se_b,
        "Sxx": Sxx, "Sxy": Sxy, "Syy": Syy,
        "n": n,
        "x_mean": x_mean, "y_mean": y_mean
    }


results = {}

for col in predictoras:
    data = df[[col, respuesta]].dropna()
    x = data[col].values.astype(float)
    y = data[respuesta].values.astype(float)

    res = simple_linear_regression(x, y)
    results[col] = res

    print("\n" + "="*60)
    print(f"Predictor: {col}")
    print(f"N (observaciones): {res['n']}")
    print(f"x̄ = {res['x_mean']:.6f}, ȳ = {res['y_mean']:.6f}")
    print(f"Sxx = Σ(x - x̄)^2 = {res['Sxx']:.6f}")
    print(f"Sxy = Σ(x - x̄)(y - ȳ) = {res['Sxy']:.6f}")
    print(f"Syy = Σ(y - ȳ)^2 = {res['Syy']:.6f}")
    print(f"Pendiente (b) = Sxy / Sxx = {res['b']:.6f}")
    print(f"Intercepto (a) = ȳ - b * x̄ = {res['a']:.6f}")
    print(f"SSE = {res['SSE']:.6f}")
    print(f"SST = {res['SST']:.6f}")
    print(f"SSR = {res['SSR']:.6f}")
    print(f"R^2 = {res['R2']:.6f}")
    print(f"RMSE = {res['RMSE']:.6f}  (error estándar del ajuste)")
    print(f"se(b) = {res['se_b']:.6f}  (error estándar de la pendiente)")
    # opcional: t-stat de la pendiente
    if not np.isnan(res['se_b']):
        t_b = res['b'] / res['se_b']
        print(f"t-stat (b / se(b)) = {t_b:.6f}")
    print("="*60)


n_preds = len(predictoras)
cols = 2
rows = int(np.ceil(n_preds / cols))

plt.figure(figsize=(10, 5 * rows))
for i, col in enumerate(predictoras, start=1):
    ax = plt.subplot(rows, cols, i)
    data = df[[col, respuesta]].dropna()
    x = data[col].values.astype(float)
    y = data[respuesta].values.astype(float)

    res = results[col]
    # Scatter
    ax.scatter(x, y, alpha=0.5, s=20, label='Datos')
    # Recta de regresión
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = res['a'] + res['b'] * x_line
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f"y = {res['a']:.3f} + {res['b']:.3f} x")
    # Información en el plot
    ax.set_xlabel(col)
    ax.set_ylabel(respuesta)
    ax.set_title(f"{col} vs {respuesta}  (R²={res['R2']:.4f})")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
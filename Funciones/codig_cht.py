# regresion_simple_sin_libs.py
import pandas as pd
import numpy as np
from math import sqrt
#from scipy.stats import t  # solo para obtener t_crit (quantil)
from pathlib import Path

# Parámetros
alpha = 0.05  # nivel 95%
predictoras = ["radio_promedio", "perimetro_promedio", "concavidad_promedio"]  # podés cambiar la lista
y_col = "area_promedio"

# Ruta al CSV (ajusta si necesario)
ruta = Path(__file__).resolve().parent.parent / "CSV" / "wdbc_full.csv"
print("Cargando:", ruta)
df = pd.read_csv(ruta)

# Omitir id/diagnostico
df = df.drop(columns=["id", "diagnostico"], errors="ignore")

n = len(df)
print(f"N = {n}\n")

def regresion_simple(x, y, x0=None, alpha=0.05):
    n = len(x)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    x_bar = x.mean()
    y_bar = y.mean()

    Sxx = np.sum((x - x_bar)**2)
    Sxy = np.sum((x - x_bar)*(y - y_bar))
    Syy = np.sum((y - y_bar)**2)

    beta1 = Sxy / Sxx
    beta0 = y_bar - beta1 * x_bar

    y_hat = beta0 + beta1 * x
    residuals = y - y_hat

    # varianza estandarizada de los residuos
    sigma2_hat = np.sum(residuals**2) / (n - 2)
    sigma_hat = sqrt(sigma2_hat)

    # errores estandar
    se_beta1 = sqrt(sigma2_hat / Sxx)
    se_beta0 = sqrt(sigma2_hat * (1.0/n + x_bar**2 / Sxx))

    # R^2 y r
    SSR = np.sum((y_hat - y_bar)**2)
    SST = Syy
    R2 = SSR / SST if SST != 0 else np.nan
    r_xy = Sxy / sqrt(Sxx * Syy)

    # t critico
    df_deg = n - 2
    t_crit = t.ppf(1 - alpha/2, df_deg)

    # Intervalos para betas
    ic_beta1 = (beta1 - t_crit*se_beta1, beta1 + t_crit*se_beta1)
    ic_beta0 = (beta0 - t_crit*se_beta0, beta0 + t_crit*se_beta0)

    # Predicción en x0 (por defecto x0 = x_bar)
    if x0 is None:
        x0 = x_bar
    y0_hat = beta0 + beta1 * x0

    se_mean = sqrt(sigma2_hat * (1.0/n + (x0 - x_bar)**2 / Sxx))
    icm_y = (y0_hat - t_crit*se_mean, y0_hat + t_crit*se_mean)

    se_pred = sqrt(sigma2_hat * (1.0 + 1.0/n + (x0 - x_bar)**2 / Sxx))
    ip_y = (y0_hat - t_crit*se_pred, y0_hat + t_crit*se_pred)

    # devolver todo en diccionario
    return {
        "n": n,
        "beta1": beta1,
        "beta0": beta0,
        "sigma2_hat": sigma2_hat,
        "sigma_hat": sigma_hat,
        "se_beta1": se_beta1,
        "se_beta0": se_beta0,
        "R2": R2,
        "r": r_xy,
        "IC_beta1": ic_beta1,
        "IC_beta0": ic_beta0,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "x0": x0,
        "y0_hat": y0_hat,
        "ICM_y": icm_y,
        "IP_y": ip_y,
        "t_crit": t_crit,
        "df": df_deg
    }

# Ejecutar para cada predictora y mostrar resultados
y = df[y_col]
resultados = {}
for xcol in predictoras:
    print("==============================================")
    print("Predictor:", xcol)
    x = df[xcol]
    res = regresion_simple(x, y, x0=None, alpha=alpha)
    resultados[xcol] = res

    # Impresion resumida (podés formatear mejor si querés)
    print(f"β1 (pendiente): {res['beta1']:.6f}")
    print(f"β0 (intercepto): {res['beta0']:.6f}")
    print(f"σ̂^2: {res['sigma2_hat']:.6f}")
    print(f"σ̂: {res['sigma_hat']:.6f}")
    print(f"R^2: {res['R2']:.6f}")
    print(f"r: {res['r']:.6f}")
    print(f"IC(β1) 95%: ({res['IC_beta1'][0]:.6f}, {res['IC_beta1'][1]:.6f})")
    print(f"IC(β0) 95%: ({res['IC_beta0'][0]:.6f}, {res['IC_beta0'][1]:.6f})")
    print(f"Predicción en x̄ = {res['x_bar']:.6f}: ŷ = {res['y0_hat']:.6f}")
    print(f"ICM(Y) en x̄ (95%): ({res['ICM_y'][0]:.6f}, {res['ICM_y'][1]:.6f})")
    print(f"IP(Y) en x̄ (95%): ({res['IP_y'][0]:.6f}, {res['IP_y'][1]:.6f})")
    print("==============================================\n")

# Si querés exportar tabla a CSV
tabla = []
for xcol, res in resultados.items():
    tabla.append({
        "predictor": xcol,
        "beta1": res['beta1'],
        "beta0": res['beta0'],
        "sigma2_hat": res['sigma2_hat'],
        "R2": res['R2'],
        "r": res['r'],
        "IC_beta1_low": res['IC_beta1'][0],
        "IC_beta1_high": res['IC_beta1'][1],
        "IC_beta0_low": res['IC_beta0'][0],
        "IC_beta0_high": res['IC_beta0'][1],
        "ICM_y_low": res['ICM_y'][0],
        "ICM_y_high": res['ICM_y'][1],
        "IP_y_low": res['IP_y'][0],
        "IP_y_high": res['IP_y'][1],
    })
pd.DataFrame(tabla).to_csv(Path(__file__).resolve().parent.parent / "CSV" / "resumen_regresion_simple.csv", index=False)
print("Resumen guardado en CSV/resumen_regresion_simple.csv")

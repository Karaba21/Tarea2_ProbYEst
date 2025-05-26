import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import numpy as np
import os

# Configuración inicial
sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# ========== Cargar datos ==========
data = pd.read_csv('cancelaciones.csv')
cancelaciones = data['cancelaciones']

# ========== Parte 1.1: Tabla de frecuencias ==========
def crear_tabla_frecuencias(cancelaciones):
    frec_abs = cancelaciones.value_counts().sort_index()
    prob_emp = frec_abs / len(cancelaciones)
    dist_acum = prob_emp.cumsum()
    
    tabla = pd.DataFrame({
        'Cancelaciones': frec_abs.index,
        'Frecuencia Absoluta': frec_abs.values,
        'Probabilidad Empírica': prob_emp.values,
        'Distribución Acumulada': dist_acum.values
    })
    return tabla

tabla_frec = crear_tabla_frecuencias(cancelaciones)
tabla_frec.to_csv('resultados/tablas/tabla_frecuencias.csv', index=False)

# ========== Parte 1.2: Medidas estadísticas ==========
def calcular_medidas(cancelaciones):
    medidas = {
        'Media': cancelaciones.mean(),
        'Varianza': cancelaciones.var(ddof=0),
        'Mediana': cancelaciones.median(),
        'Q1': cancelaciones.quantile(0.25),
        'Q3': cancelaciones.quantile(0.75),
        'IQR': cancelaciones.quantile(0.75) - cancelaciones.quantile(0.25)
    }
    return pd.Series(medidas)

medidas = calcular_medidas(cancelaciones)
print("\nMedidas estadísticas:")
print(medidas.to_string())

# ========== Parte 1.3: Gráficos ==========
def generar_graficos(cancelaciones):
    # Histograma
    plt.figure(figsize=(10, 5))
    plt.hist(cancelaciones, bins=range(2, 16), density=True, alpha=0.7, edgecolor='black')
    plt.title('Histograma de cancelaciones diarias')
    plt.xlabel('Número de cancelaciones')
    plt.ylabel('Densidad')
    plt.savefig('resultados/graficos/histograma.png')
    plt.close()
    
    # Diagrama de caja
    plt.figure(figsize=(8, 4))
    plt.boxplot(cancelaciones, vert=False, patch_artist=True)
    plt.title('Diagrama de caja de cancelaciones')
    plt.xlabel('Cancelaciones diarias')
    plt.savefig('resultados/graficos/boxplot.png')
    plt.close()
    
    # Ajuste Poisson
    x = range(2, 16)
    poisson_pmf = poisson.pmf(x, mu=cancelaciones.mean())
    
    plt.figure(figsize=(10, 5))
    plt.hist(cancelaciones, bins=range(2, 16), density=True, alpha=0.7, label='Datos')
    plt.plot(x, poisson_pmf, 'ro-', label=f'Poisson (λ={cancelaciones.mean():.2f})')
    plt.title('Ajuste a distribución de Poisson')
    plt.xlabel('Cancelaciones diarias')
    plt.ylabel('Densidad')
    plt.legend()
    plt.savefig('resultados/graficos/poisson_ajuste.png')
    plt.close()

generar_graficos(cancelaciones)

# ========== Parte 1.4: Probabilidades con Poisson ==========
def prob_poisson(cancelaciones):
    lambda_ = cancelaciones.mean()
    p_menos_5 = poisson.cdf(4, mu=lambda_)
    p_mas_15 = 1 - poisson.cdf(15, mu=lambda_)
    
    print(f"\nProbabilidades con Poisson (λ={lambda_:.2f}):")
    print(f"P(X < 5): {p_menos_5:.4f}")
    print(f"P(X > 15): {p_mas_15:.4f}")

prob_poisson(cancelaciones)
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, cauchy

def generar_uniformes(n, a, c, m, semilla=None):
    muestras = []
    if semilla is None:
        semilla = int(time.time()) % m
    x = semilla
    for i in range(n):
        x = (a * x + c) % m
        muestras.append(x / m)  # intervalo [0, 1)
    return muestras

def generar_cauchy(n, a, c, m, semilla=None):
    uniformes = generar_uniformes(n, a, c, m, semilla)
    
    # Aplicar transformación inversa para obtener Cauchy estándar
    cauchy_samples = [np.tan(np.pi * (u - 0.5)) for u in uniformes]
    
    return cauchy_samples

# Parámetros recomendados
a = 1664525
c = 1013904223
m = 2**32

# Generar 100 muestras
muestras_cauchy = generar_cauchy(100, a, c, m)

# Configuración del gráfico
plt.figure(figsize=(12, 6))

# Histograma de las muestras de Cauchy
sns.histplot(muestras_cauchy, bins=100, stat='density', alpha=0.5, 
             color='skyblue', edgecolor='black', label='Histograma',
             binrange=(-10, 10))

# Estimación de densidad por núcleos (KDE)
kde = gaussian_kde(muestras_cauchy)
x_vals = np.linspace(-10, 10, 1000)
plt.plot(x_vals, kde(x_vals), color='red', linewidth=2, label='Estimacion de Densidad por Núcleos (KDE)')

# PDF (funcion de probabilidad de densidad) teórica de Cauchy estándar
plt.plot(x_vals, cauchy.pdf(x_vals), color='green', linewidth=2, 
         linestyle='--', label='PDF teórica Cauchy')

# Formato del gráfico
plt.title('Distribución de Cauchy Estándar Simulada', fontsize=14)
plt.xlabel('Valor de la muestra', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show()
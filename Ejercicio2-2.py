import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def generar_uniformes(n, a, c, m, semilla=None):
    muestras = []
    if semilla is None:
        semilla = int(time.time()) % m
    x = semilla
    for i in range(n):
        x = (a * x + c) % m
        muestras.append(x / m)  # intervalo [0, 1)
    return muestras

# Parámetros recomendados
a = 1664525
c = 1013904223
m = 2**32

# Generar 100 muestras
muestras = generar_uniformes(100, a, c, m)

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Histograma
sns.histplot(muestras, bins=10, stat='density', alpha=0.5, color='skyblue', edgecolor='black', label='Histograma')

# Estimación de densidad por núcleos (KDE)
kde = gaussian_kde(muestras)
x_vals = np.linspace(min(muestras), max(muestras), 1000)
plt.plot(x_vals, kde(x_vals), color='red', linewidth=2, label='KDE')

# Formato del gráfico
plt.title('Histograma y Estimación de Densidad por Núcleos', fontsize=14)
plt.xlabel('Valor de la muestra', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show()
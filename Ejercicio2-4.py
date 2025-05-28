import time
import numpy as np

def generar_uniformes(n, a, c, m, semilla=None):
    muestras = []
    if semilla is None:
        semilla = int(time.time()) % m
    x = semilla
    for _ in range(n):
        x = (a * x + c) % m
        muestras.append(x / m)
    return muestras

def generar_cauchy(n, a, c, m, semilla=None):
    uniformes = generar_uniformes(n, a, c, m, semilla)
    return [np.tan(np.pi * (u - 0.5)) for u in uniformes]

# Parámetros recomendados
a = 1664525
c = 1013904223
m = 2**32

# Generar una muestra
muestra_cauchy = generar_cauchy(1, a, c, m)

print(f"Muestra de Cauchy estándar generada: {muestra_cauchy}")
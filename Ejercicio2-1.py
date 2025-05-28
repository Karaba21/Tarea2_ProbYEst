import time
import numpy as np


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

# Generar muestra
muestras = generar_uniformes(1, a, c, m)
print(f"Muestra generada: {muestras}")
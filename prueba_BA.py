"""
Prueba del Bosque Aleatorio
Miniproyecto 3
"""

import random
from utileria import descarga_datos, descomprime_zip, lee_csv
from bosque_aleatorio import entrena_bosque, evalua_bosque

# ======================================================
# 1. Descarga y carga del dataset
# ======================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
archivo = "wdbc.data"

print("Descargando dataset...")
descarga_datos(url, archivo)

atributos = [
    "id",
    "diagnosis",
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

datos = lee_csv(archivo, atributos=atributos)

# Convertir a numéricos excepto diagnosis
for d in datos:
    for k in d:
        if k not in ["diagnosis"]:
            d[k] = float(d[k])

# ======================================================
# 2. Separar entrenamiento y prueba
# ======================================================

random.shuffle(datos)

split = int(0.8 * len(datos))
train = datos[:split]
test = datos[split:]

target = "diagnosis"
clase_default = "M"

print("Total datos:", len(datos))
print("Entrenamiento:", len(train))
print("Prueba:", len(test))

# ======================================================
# 3. Experimento 1: Número de árboles
# ======================================================

print("\n=== Experimento: Número de árboles ===")

for n_arboles in [1, 5, 10, 20, 50]:
    bosque = entrena_bosque(
        train,
        target,
        clase_default,
        n_arboles=n_arboles,
        max_profundidad=5,
        variables_seleccionadas=5
    )

    acc = evalua_bosque(bosque, test, target)
    print(f"Árboles: {n_arboles} -> Accuracy: {acc:.4f}")

# ======================================================
# 4. Experimento 2: Profundidad máxima
# ======================================================

print("\n=== Experimento: Profundidad máxima ===")

for profundidad in [1, 3, 5, 10, None]:
    bosque = entrena_bosque(
        train,
        target,
        clase_default,
        n_arboles=20,
        max_profundidad=profundidad,
        variables_seleccionadas=5
    )

    acc = evalua_bosque(bosque, test, target)
    print(f"Profundidad: {profundidad} -> Accuracy: {acc:.4f}")

# ======================================================
# 5. Experimento 3: Variables por nodo
# ======================================================

print("\n=== Experimento: Variables por nodo ===")

for vars_nodo in [1, 3, 5, 10, 20]:
    bosque = entrena_bosque(
        train,
        target,
        clase_default,
        n_arboles=20,
        max_profundidad=5,
        variables_seleccionadas=vars_nodo
    )

    acc = evalua_bosque(bosque, test, target)
    print(f"Variables nodo: {vars_nodo} -> Accuracy: {acc:.4f}")
import random
from collections import Counter
from arboles_numericos import entrena_arbol

def entrena_bosque(datos, target, clase_default,
                   n_arboles=10,
                   max_profundidad=None,
                   acc_nodo=1.0,
                   min_ejemplos=0,
                   variables_seleccionadas=None):
    """
    Entrena un bosque aleatorio.
    Regresa una lista de árboles (raíces).
    """
    bosque = []

    for _ in range(n_arboles):

        n = len(datos)
        subconjunto = [random.choice(datos) for _ in range(n)]

        arbol = entrena_arbol(
            subconjunto,
            target,
            clase_default,
            max_profundidad,
            acc_nodo,
            min_ejemplos,
            variables_seleccionadas
        )

        bosque.append(arbol)

    return bosque


def predice_bosque(bosque, instancia):
    """
    Predice la clase de una instancia usando votación mayoritaria.
    """
    predicciones = [arbol.predice(instancia) for arbol in bosque]
    conteo = Counter(predicciones)
    return conteo.most_common(1)[0][0]


def predice_bosque_lista(bosque, datos):
    """
    Predice una lista de instancias.
    """
    return [predice_bosque(bosque, d) for d in datos]


def evalua_bosque(bosque, datos, target):
    """
    Evalúa el accuracy del bosque.
    """
    predicciones = predice_bosque_lista(bosque, datos)
    return sum(1 for p, d in zip(predicciones, datos)
               if p == d[target]) / len(datos)
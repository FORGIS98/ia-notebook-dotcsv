#!/usr/bin/env python
# coding: utf-8

import numpy as np # Para calculo numérico, calculos con array y operaciones con matrices...
import scipy as sc # Extiende numpy a herramientas científicas (tratamiento de imágenes y datos de otras maneras)

import matplotlib.pyplot as plt # Librería para visualizar datos en gráficos y demás

# Los datasets se pueden conseguir de varias maneras,
# puede ser un CSV que descargues de internet. Para este ejemplo,
# utilizaremos el dataset de boston que tiene la librería sklearn.
from sklearn.datasets import load_boston

# Cargamos la librería
boston = load_boston()

# print(boston.DESCR) # Para ver info del dataset


# Formula de minimizar el error cuadrático medio(MCO): $\beta = (X^{T}X)^{-1}X^{T}Y$  
# Siendo `X` matriz_entrada e `Y` valor_medio


# np.array() transforma los datos a array
matriz_entrada = np.array(boston.data[:, 5]) # Todas las filas de la columna 5 (que es la de RM = average nº room)
valor_medio = np.array(boston.target) # valor medio de la vivienda, esta en target y no en data por que why not

# Mostrar los datos en una gráfica
plt.scatter(matriz_entrada, valor_medio, alpha=0.3) # Ejes X,Y y alpha=transparencia

# Añadimos columna de 1's para el termino independiente
matriz_entrada = np.array([np.ones(len(matriz_entrada)), matriz_entrada]).T # Traspuesta

# Formula del error cuadrático medio
# La T es de traspuesta, el @ es por que nos interesa multiplicación matricial (el * es multiplicación escalar)
# np.linalg.inv() es la inversa de lo que hay en los ()
beta = np.linalg.inv(matriz_entrada.T @ matriz_entrada) @ matriz_entrada.T @ valor_medio
# Ahora tenemos nuestra matriz que minimizan el error cuadrático medio de nuestra nube de puntos.
# beta = [valor en el que corta cuando X es igual a 0, pendiente]

plt.plot([4, 9], [beta[0] + beta[1] * 4, beta[0] + beta[1] * 9], c="red") # Linea de regresión lineal estimada por los mínimos cuadrados ordinarios
plt.show()

# Es bonito haber escrito la formula del error cuadrático medio, pero es raro hacerlo pues las librerías
# ya traen dicha formula de serie. Hay métodos mas bonitos como el descenso del gradiente que esta en IA Notebook #3
# Lo especial del descenso del gradiente a diferencia del MCO es que el MCO da exáctamente el mínimo error de una,
# y señor gradiente va acercandose poco a poco al minimo error de manera iterativa. 


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:48:21 2017

@author: david
"""


from pyspark import SparkContext
import numpy as np
from time import time


class KMeansSpark():
    
    def __init__(self, k, epsilon=1, max_iter=10):
        self.k = k # numero de clusters
        self.epsilon = epsilon # tolerancia para el criterio de parada
        self.max_iter = max_iter # maximo numero de iteraciones
        
    def distance_squared(self, p, q): 
        # Distancia al cuadrado entre dos puntos
        return np.sum((p - q)**2)
    
    def closest_centroid(self, p):
        # para un punto p devuelve el indice del centroide mas proximo a p
        index = np.argmin(np.linalg.norm(self.centroids - p, axis=1))
        return index

    def fit(self, X): # X es un RDD de np.array
        # ajusta los centroides a los datos
        init_time = time()
        X.persist() # solo si tenemos suficiente memoria
        # se empieza con k puntos del dataset eleccionados aleatoriamente
        initial_centroids = np.asarray(X.takeSample(False, self.k, 42))
        self.centroids = initial_centroids
        
        distance = np.inf
        it = 0 # iteracion
        while (distance > self.epsilon and it < max_iter):
        
            # Para cada punto, encontrar el indice del centroide mas proximo
            # mapearlo a (index, (point,1))
            points_clusterized = X.map(lambda p: ( self.closest_centroid(p), \
                                                  (p, 1) ))
                
            # para cada key (k-point index), hacer una agregacion
            # de las coordenadas y el numero de puntos
            clusters_set = points_clusterized\
                       .reduceByKey(lambda p1, p2: (p1[0]+p2[0], p1[1]+p2[1]) )
            
            # para cada key (k-point index), encontrar los nuevso centroides
            # calculando la media de los puntos de un mismo cluster (centroid)
            new_centroids = clusters_set\
                  .mapValues(lambda pair: (pair[0] / pair[1]) )\
                  .sortBy(lambda t: t[0])\
                  .map(lambda pair: pair[1]).collect()
            new_centroids = np.asarray(new_centroids)
            
            # calculamos la distancia entre los nuevos centroides y los anteriores
            distance = np.linalg.norm(self.centroids - new_centroids, ord=np.inf)
            
            #Asignamos los nuevos centroides al array de centroides de la clase
            self.centroids = new_centroids
            it += 1
        time_passed = time() - init_time
        if it == max_iter:
            print('Maximun number of iterations reached')
            print('{} iterations terminated in {} seconds'.format(it, time_passed))
        else:
            print('Convergence successfull')
            print('{} iterations terminated in {} seconds'.format(it, time_passed))

if __name__ == '__main__':
    
    # master puede ser 'local[numero_de_hilos]', 'yarn-client', 'yarn-cluster'
    master = 'local[*]'
    sc = SparkContext(appName='KMeansSpark', master=master)
    k = 4
    epsilon = 0.1
    max_iter = 10
    
    from_file = False
    
    # Testarlo localmente
    if not from_file:
        np.random.seed(42)
        D, N = 2, 150
        mu0 = np.array([1, 3])
        X0 = (np.random.randn(D, N) + mu0[:, np.newaxis]).T
        mu1 = np.array([7, 7])
        X1 = (np.random.randn(D, N) + mu1[:, np.newaxis]).T
        mu2 = np.array([2, -5])
        X2 = (np.random.randn(D, N) + mu2[:, np.newaxis]).T
        mu3 = np.array([-5, -1])
        X3 = (np.random.randn(D, N) + mu3[:, np.newaxis]).T 
        X = np.vstack((X0, X1, X2, X3))
        X = sc.parallelize(X)
    else:
        number_partitions = 2
        sep = " "
        X = sc.textFile('/path/to/your/textfile', number_partitions)\
              .map(lambda s: np.fromstring(s, dtype=np.float64, sep=sep))
    
    k_means = KMeansSpark(k, epsilon)
    k_means.fit(X)
    
    for centroid in k_means.centroids:
        print(centroid)

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 08:56:40 2017

@author: david
"""

from __future__ import print_function, division
import numpy as np
from pyspark import SparkContext
 

class LinearRegression():

    def __init__(self, alpha=1, tolerance=5, max_steps=10):
        self.theta = None
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_steps = max_steps
        
    def map_function(self, t):
        x, y = t
        error = self.theta.T.dot(x) - y
        return (error**2, error * x)
    
    def reduce_function(self, t1, t2):
        error_squared_1, error_dot_x_1 = t1[0], t1[1]
        error_squared_2, error_dot_x_2 = t2[0], t2[1]
        return(error_squared_1+error_squared_2, error_dot_x_1+error_dot_x_2)
        
    def fit(self, rdd): # rdd = x; y
        tuples_x_y = rdd.map(lambda arr: (arr[0:-1], arr[-1]))
        tuples_x_y.persist()
        
        m = tuples_x_y.count() # numero de ejemplos de entrenamiento
        n = tuples_x_y.first()[0].shape[0] # numero de caracteristicas
        self.theta = np.random.rand(n, 1) # Inicializamos theta aleatoriamente
        J = np.inf
        steps = 0
        
        while J > self.tolerance and steps < self.max_steps:
            errors = tuples_x_y.map(self.map_function)
            sum_errors = errors.reduce(self.reduce_function)
            print(sum_errors)
            sum_root_error = sum_errors[0][0]
            root_error_dot_x = sum_errors[1].reshape((-1,1))
            print("sum_root_error", sum_root_error)
            print("root_error_dot_x", root_error_dot_x)
            J = (1 / (2*m)) * sum_root_error
            self.theta = self.theta - (self.alpha / m) * root_error_dot_x
            steps += 1
            #print('Iteracion: ', steps, ' ; Error: ', J)
            
if __name__ == '__main__':
    
    sc = SparkContext()
    
    sep = ','
    alpha = 1
    tolerance = 5
    max_steps = 10
    lr = LinearRegression(alpha, tolerance, max_steps)
    
    data = sc.textFile("file:///home/training/Desktop/data3.txt")\
             .map(lambda s: np.fromstring(s, dtype=np.float64, sep=sep))
    lr.fit(data)
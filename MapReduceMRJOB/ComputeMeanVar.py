# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 13:21:43 2017

@author: David Retana

Computacion de manera paralela
de la media y la desviacion tipica
"""

from __future__ import print_function, division
from mrjob.job import MRJob
import sys


class ComputeMeanVar(MRJob):
    
    regex = ","

    def mapper(self, _, line):
       fields = line.split(self.regex)
       for i in range(len(fields)):
           yield(i, (float(fields[i]), 1))

    def reducer(self, key, values):
       m = 0.0 # numero de registros
       sum_features = 0.0
       sum_features_squared = 0.0
       for feature, i in values:
           sum_features += feature
           sum_features_squared += feature**2
           m += i
       muj = sum_features / m
       sj2 = (sum_features_squared + m*muj**2 -2*muj*sum_features) / m
       yield (key, (muj, sj2))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage computeMeanVar: <input_file>', file=sys.stderr)
        exit(-1)
    print('Starting parallelized computation of mean and var')
    path = sys.argv[1]
    job = ComputeMeanVar(args=[path])
    runner = job.make_runner()
    runner.run()
    tmp_output = []
    for line in runner.stream_output(): #stream_output es un generador
        tmp_output.append(line.split("\t"))
    for i in tmp_output:
        print('element: ', i)
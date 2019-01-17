import sys
import csv
import numpy as np

from Algorithm import update_algorithm
from Algorithm import get_closest
from Algorithm import stabilised_solution
from graphic import display

from pyspark import SparkContext



if len(sys.argv) != 3:
    print("usage : python helloworld.py name_of_file_points number_of_cluster")
    sys.exit(1)

log_file = sys.argv[1]  # Should be some file on your system
nb_centroid = int(sys.argv[2])
sc = SparkContext("local", "K-Means App")

with open(log_file, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)

# number of iterations
k = 10
# stop criteria
stop_criteria = 0.00001

with open(log_file) as inputfile:
    next(inputfile)
    reader = csv.reader(inputfile)
    inputm = list(reader)

float_points = list(np.float_(inputm))
float_centroids = []
for item in range(nb_centroid):
    float_centroids.append(float_points[np.random.randint(len(float_points))])
map_points = sc.parallelize(float_points)
map_centroid = sc.parallelize(float_centroids)

map_centroid_old = map_centroid
print("at iteration 0 : \n")
print(map_centroid.collect())
for x in range(0, k):
    if x != 0 and stabilised_solution(map_centroid_old, map_centroid, nb_centroid, stop_criteria):
        break
    else:
        map_centroid_old = map_centroid
    list_closest = get_closest(sc, map_points, map_centroid)
    map_centroid = update_algorithm(map_points, list_closest, nb_centroid, sc, map_centroid)
    display(map_points, list_closest, log_file)
    print("------------------------------------------------")
    print("at iteration ")
    print(x + 1)
    print(": \n")
    print(map_centroid.collect())

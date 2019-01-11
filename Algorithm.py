import numpy as nump
from scipy import spatial


def update_algorithm(points, list_closest, nb_c, sc, map_centroid):
    point = points.collect()
    closest = list_closest.collect()
    cluster = map_centroid.collect()
    nb_coord = len(point[0])
    nb_points = len(closest)

    for x in range(0, nb_c):
        result_sum = [0.0] * nb_coord
        result_card = 0
        for l in range(0, nb_points):
            if closest[l] == x:
                for coord in range(0, nb_coord):
                    result_sum[coord] += point[l][coord]
                result_card += 1
        if result_card != 0:
            for coord in range(0, nb_coord):
                result_sum[coord] /= result_card
            cluster[x] = result_sum

    map_result = sc.parallelize(cluster)
    return map_result


def get_closest(sc, points, centers):
    point = points.collect()
    center = centers.collect()

    dist_matrix = spatial.distance_matrix(point, center)
    result_id = nump.array([dist_matrix.argmin(axis=1)])
    result = sc.parallelize(result_id[0])
    return result


def stabilised_solution(old_points, new_points, nb, stop_criteria):
    old = old_points.collect()
    new = new_points.collect()

    dist_matrix = spatial.distance_matrix(old, new)
    for x in range(0, nb):
        if dist_matrix[x][x] > stop_criteria:
            return False
    return True


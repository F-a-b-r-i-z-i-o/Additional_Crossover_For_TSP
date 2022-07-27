###################################################
# The supporting libraries used are:
# numpy -  To generate random integers as indices
# random - To shuffle the list of integers
# copy - To deepcopy a list
# time - To generate seeds for random numbers generator
# operator - To sort the dictionary
###################################################

from Ga_algo import *
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
from loguru import logger
logger.add("logs_file.log")


# Main Class which implements the GA algorithm which takes as input the parameters tsp_len, pop_size, distance matrix, iterations, elitism flag, crossover_type


class File:
    def __init__(self):

        self.tsp_len = None

        self.iterations = None

        self.elitism = None

        self.pop_size = None

        self.file_name = None

        self.crossover = None

    @hydra.main(config_name="config.yaml")
    def configsetters(cfg):
        init_values = []
        final_values = []
        ans_values = []

        tsp_len = cfg.main.tsp_len
        iterations = cfg.main.iterations
        elitism = cfg.main.elitism
        pop_size = cfg.main.pop_size
        file_name = cfg.main.file_name
        crossover = cfg.main.crossover
        logger.info(cfg.main)
        cwd = os.getcwd()
        with open(file_name, 'r') as fp:
            data = fp.readlines()
        data = [[int(j) for j in i.replace("\n", "").split(',')] for i in data]
        original_points = data
        pop_size = len(data)
        weights = np.zeros((pop_size, pop_size), dtype=np.float64)

        for i in range(pop_size):
            for j in range(pop_size):
                weights[i][j] = (original_points[i][0] - original_points[j]
                                 [0])**2 + (original_points[i][1] - original_points[j][1])**2
                weights[i][j] = weights[i][j]**0.5
        obj = GAalgo(tsp_len, pop_size, weights,
                     iterations, elitism, crossover)
        init_values.append(1/obj.cost(obj.population[0]))
        val, ans = obj.run_algo()
        print(val)
        ans_values.append(ans)
        final_values.append(val)
        logger.info(ans_values)
        logger.info(final_values)

        # Fetching the best solution
        pts = np.array(original_points)
        pts = pts[ans_values[0]]
        joining_pts = np.zeros((2, 2))
        joining_pts[0] = pts[-1]
        joining_pts[1] = pts[0]
        plt.title("Best Solution Tour using select Crossover")

        plt.plot(pts[0][0], pts[0][1], color='orange', marker='o',
                 linestyle='dashed', label="Starting city")
        plt.plot(pts[1:, 0], pts[1:, 1], color='green',
                 marker='o', linestyle='dashed')
        plt.plot(pts[-1][0], pts[-1][1], color='blue', marker='o',
                 linestyle='dashed', label="Ending city")
        plt.plot(joining_pts[:, 0], joining_pts[:, 1],
                 color='orange', linestyle='dashed')
        plt.legend()
        plt.show()


# Running experiments
if __name__ == "__main__":
    File.configsetters()
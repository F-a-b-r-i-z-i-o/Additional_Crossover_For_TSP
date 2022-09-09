###################################################
# Libraries used are:
# numpy -  To generate random integers as indices
# matplotlib - To plot a graph
# hydra - Create a hierarchical configuration
# loguru - library which aims to bring enjoyable logging in Python.
###################################################

from Ga_algo import *
import matplotlib.pyplot as plt
import numpy as np
import os
from hydra import compose, initialize
from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import argparse

logger.add('../Result/result.log')
file_name = ('../Test/bruma14.txt')


# File Class which implements the GA algorithm which takes as input the parameters tsp_len, pop_size, distance matrix, iterations, elitism flag, crossover_type

def configsetters(cfg, plot=False):
    """
        Read configuration of algoritm by the file.yaml
        Plot the path of algoritm
        Writes the best path and the best solution in the log file
    """

    # List of init value
    init_values: float
    ans_values: list
    final_value: float

    """
        Read the configuration by the file.yaml
    """

    tsp_len = cfg.main.tsp_len
    iterations = cfg.main.iterations
    elitism = cfg.main.elitism
    pop_size = cfg.main.pop_size
    best_n = cfg.main.pop_size
    file_name = cfg.main.file_name
    crossover = cfg.main.crossover
    logger.info(cfg.main)

    split_file_name = Path(file_name).parts

    istances = split_file_name[7]

    cwd = os.getcwd()

    with open(file_name, 'r') as fp:
        data = fp.readlines()

    data = [[float(j) for j in i.replace("\n", "").split(',')]
            for i in data]

    # Create original data
    original_points = data

    pop_size = len(data)

    # Create the weights matrix
    weights = np.zeros((pop_size, pop_size), dtype=np.float64)

    # Calc eucliedean distance of data
    for i in range(pop_size):
        for j in range(pop_size):
            weights[i][j] = (original_points[i][0] - original_points[j]
                             [0])**2 + (original_points[i][1] - original_points[j][1])**2
            weights[i][j] = weights[i][j]**0.5

    # Application of GA_ALGORITM
    obj = GAalgo(tsp_len, pop_size, weights,
                 iterations, elitism, crossover, best_n)

    # Find init value
    init_value = 1/obj.cost(obj.population[0])

    final_value, ans_values = obj.run_algo()

    # Print best value select
    print(final_value)

    # Save log of cycle
    logger.info(ans_values)

    # Save log of best value
    logger.info(final_value)

    d = {'istances': istances,
         'geneation': iterations,
         'crossover': crossover,
         'population': pop_size,
         'final_value': final_value
         }

    if plot:
        # Plot cost graph
        obj.graph()

        # Fetching the best solution
        pts = np.array(original_points)
        pts = pts[ans_values]

        joining_pts = np.zeros((2, 2))
        joining_pts[0] = pts[-1]
        joining_pts[1] = pts[0]

        # Plot graph of city path
        plt.title("Solution Tour crossover: " + crossover)

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

    return d

    # Running experiments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations',
                        help='number of iterations', type=int)
    parser.add_argument(
        '-p', '--plot', help='plot fitness function and solution tour', action='store_true')

    # python3 main.py -o ../Result/ -i 2

    parser.add_argument(
        '-o', '--output', help="the csv's output path", type=str)

    args = parser.parse_args()

    initialize(version_base=None, config_path="./", job_name="tsp")
    cfg = compose(config_name="config.yaml")

    d = {
        'Istances': [],
        'Geneation': [],
        'Crossover': [],
        'Population': [],
        'Final Value': []
    }

    for i in range(args.iterations):
        df = configsetters(cfg, plot=args.plot)

        d['Istances'].append(df['istances'])
        d['Geneation'].append(df['geneation'])
        d['Crossover'].append(df['crossover'])
        d['Population'].append(df['population'])
        d['Final Value'].append(df['final_value'])

    df = pd.DataFrame(data=d)

    df.to_csv(os.path.join(args.output, 'analisys.csv'), index=False, encoding='utf-8',
              escapechar='\t', mode='a+')

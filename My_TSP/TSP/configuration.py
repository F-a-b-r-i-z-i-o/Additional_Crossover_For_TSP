###################################################
# Libraries used are:
# numpy -  To generate random integers as indices
# matplotlib - To plot a graph
# hydra - Create a hierarchical configuration
# loguru - library which aims to bring enjoyable logging in Python.
###################################################

import statistics
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
import csv
from statistics import mean


class Configuration:

    logger.add('../Log/result.log')

    file_name = ('../Istances/dantzig42.txt')

    def __init__(self):

        self.d_mean = {
            'Mean': []
        }

        self.d = {
            'Istances': [],
            'Geneation': [],
            'Crossover': [],
            'Population': [],
            'Final Value': [],
        }

        self.temp = []
        self.average = []

        # List of init value
        self.init_values: float
        self.ans_values: list
        self.final_value: float

        self.create_analys()

    def configsetters(self, cfg, plot=False):
        """
            Read configuration of algoritm by the file.yaml
            Plot the path of algoritm
            Writes the best path and the best solution in the log file
        """

        """
            Read the configuration by the file.yaml
        """

        tsp_len = cfg.main.tsp_len
        iterations = cfg.main.iterations
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
        obj = GAalgo(tsp_len, weights,
                     iterations, crossover)

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
             'final_value': final_value,
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

    def create_analys(self):
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

        for i in range(args.iterations):
            df = self.configsetters(cfg, plot=args.plot)

            self.d['Istances'].append(df['istances'])
            self.d['Geneation'].append(df['geneation'])
            self.d['Crossover'].append(df['crossover'])
            self.d['Population'].append(df['population'])
            self.d['Final Value'].append(df['final_value'])

        df = pd.DataFrame(data=self.d)

        df.to_csv(os.path.join(args.output, 'analisys.csv'), index=False, encoding='utf-8',
                  escapechar='\t', mode='w')

        with open('../Result/analisys.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.temp.append(float(row[4]))
                    mean_result = statistics.mean(self.temp)
                    mean_result = round(mean_result)
                    line_count += 1
            self.average.append(mean_result)

        self.d_mean['Mean'].append(mean_result)

        df2 = pd.DataFrame(data=self.d_mean)

        df2.to_csv('../Result/analisys.csv', index=False, mode='a+')

        df2 = pd.read_csv(
            '/home/fabrizio/Scrivania/Much-Cross-Little-Over/My_TSP/Result/analisys.csv')

        print(df2.to_string(index=False))

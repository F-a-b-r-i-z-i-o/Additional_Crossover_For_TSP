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
import pandas as pd
from pathlib import Path
import argparse
import csv
from statistics import mean


class Configuration:

    # Add file .log
    logger.add('../Log/result.log')

    # Input file
    file_name = ('../Istances-Converter/gr202.txt')

    def __init__(self):

        # Dict for mean df
        self.d_mean = {
            'Mean': []
        }

        # Dict df
        self.d = {
            'Istances': [],
            'Geneation': [],
            'Crossover': [],
            'Population': [],
            'Final Value': [],
        }

        # Temp list for calc mean
        self.temp = []

        # List of mean value
        self.average = []

        # Init value define
        self.init_values: float

        # Ans value define
        self.ans_values: list

        # Final value define
        self.final_value: float

        # Recall create analys
        self.create_analys()

    def configsetters(self, cfg, plot=False):
        """
            Read configuration of algoritm by the file.yaml
            Plot the path of algoritm
            Writes the best path and the best solution and path in the log file
        """

        """
            Read the configuration by the file.yaml
        """

        tsp_len = cfg.main.tsp_len
        iterations = cfg.main.iterations
        file_name = cfg.main.file_name
        crossover = cfg.main.crossover

        logger.info(cfg.main)

        # Split input file
        split_file_name = Path(file_name).parts

        # Find .txt file extension
        istances = split_file_name[7]

        cwd = os.getcwd()

        # Open and read file input
        with open(file_name, 'r') as fp:
            data = fp.readlines()

        data = [[float(j) for j in i.replace("\n", "").split(',')]
                for i in data]

        # Create original data
        original_points = data

        # Create pop_size
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

        # Add value in dict
        d = {'istances': istances,
             'geneation': iterations,
             'crossover': crossover,
             'population': pop_size,
             'final_value': final_value,
             }

        # Control if i would plot graph
        if plot:
            # Plot cost graph
            obj.graph()

            # Fetching the best solution
            pts = np.array(original_points)

            # Crate pts variable from ans_value that is a return path
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
        '''

            Read the instruction for number of iteration and plot graph by terminal line 

            Exaple:
            Only Iteration ---> python3 main.py -i 3 -o ../Result/
            Plot Graph Whit iteration ---> python3 main.py -i 3 -p -o ../Result/

        '''

        # Define parser
        parser = argparse.ArgumentParser()

        # Add iteration with value and letter
        parser.add_argument('-i', '--iterations',
                            help='number of iterations', type=int)

        # Add plot with letter
        parser.add_argument(
            '-p', '--plot', help='plot fitness function and solution tour', action='store_true')

        # Add output with path result
        parser.add_argument(
            '-o', '--output', help="the csv's output path", type=str)

        # Recall parser
        args = parser.parse_args()

        # Initialize with a configuration path relative to the caller
        initialize(version_base=None, config_path="./", job_name="tsp")
        cfg = compose(config_name="config.yaml")

        '''
        
            Create 2 DF. 
            
            .1 Df contains the value of:
            
                istances, generation, crossover, population, final value 
                
            that change based iterations 
            
            .2 DF contains the mean of all best value 
            
            The result of each other is save on folder /Result  in file analysis.csv 
            
        '''

        for i in range(args.iterations):
            df = self.configsetters(cfg, plot=args.plot)

            # Populate df by value dict
            self.d['Istances'].append(df['istances'])
            self.d['Geneation'].append(df['geneation'])
            self.d['Crossover'].append(df['crossover'])
            self.d['Population'].append(df['population'])
            self.d['Final Value'].append(df['final_value'])

        # Create 1.DF
        df = pd.DataFrame(data=self.d)

        df.to_csv(os.path.join(args.output, 'analisys.csv'), index=False, encoding='utf-8',
                  escapechar='\t', mode='w')

        # Read the csv create and take the final value for mean
        with open('../Result/analisys.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.temp.append(float(row[4]))
                    mean_result = mean(self.temp)
                    mean_result = round(mean_result)
                    line_count += 1
            self.average.append(mean_result)

        # Inser the mean value on df
        self.d_mean['Mean'].append(mean_result)

        # Create df
        df2 = pd.DataFrame(data=self.d_mean)

        # Write df
        df2.to_csv('../Result/analisys.csv', index=False, mode='a+')

        df2 = pd.read_csv(
            '/home/fabrizio/Scrivania/Much-Cross-Little-Over/My_TSP/Result/analisys.csv')

        # Print final df
        print(df2.to_string(index=False))

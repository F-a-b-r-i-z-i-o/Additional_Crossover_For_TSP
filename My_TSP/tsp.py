###################################################
# Tejasvi Sharma
# M.Tech AI
# The entire code is done from scratch without any GA library. The supporting libraries used are:
# numpy -  To generate random integers as indices
# random - To shuffle the list of integers
# copy - To deepcopy a list
# time - To generate seeds for random numbers generator
# operator - To sort the dictionary
###################################################
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import time
import operator
import os
import hydra
from loguru import logger
logger.add("logs_file.log")
# Reading the input data
filename = "City_Coordinates.txt"

# Fixing the crossover and mutation probabilities
p_crossover = 0.95
p_mutation = 0.01

# Main Class which implements the GA algorithm which takes as input the parameters tsp_len, pop_size, distance matrix, iterations, elitism flag, crossover_type


class File:
    init_values = []
    final_values = []
    ans_values = []

    @hydra.main(config_name="config.yaml")
    def __init__(self):

        self.tsp_len = None

        self.iterations = None

        self.elitism = None

        pop_size = None

        file_name = None

        crossover = None

    def configsetters(cfg):
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


class GAalgo:
    def __init__(self, tsp_len, pop_size, weights, iterations, elitism, crossover_type):
        self.elitism = elitism
        self.iterations = iterations
        self.tsp_len = tsp_len
        self.pop_size = pop_size
        self.crossover_type = crossover_type

        elements = [i for i in range(tsp_len)]
        population = []
        p = np.random.permutation(tsp_len)

        for i in range(pop_size):
            population.append(list(np.array(elements)[p.astype(int)]))
            p = np.random.permutation(tsp_len)
        self.population = population
        self.weights = weights

    def cost(self, sol):
        value = 0.0
        weights = self.weights
        for i in range(self.tsp_len-1):
            value += weights[sol[i-1]][sol[i]]
        value += weights[sol[-1]][sol[0]]
        return 1/value

    def selection(self, res):

        sum = 0.0
        rvalue = np.random.rand()
        j = 0
        for i in res:
            sum += i[1]
            if(sum >= rvalue):
                ret = j
                break
            j += 1
        return self.population[res[j][0]]

    def pop_selection(self):
        dict = {}
        sum = 0.0
        for ind, pop in enumerate(self.population):
            val = self.cost(pop)
            dict[ind] = 1/val
        res1 = sorted(dict.items(), key=lambda i: i[1])
        dict = {}
        for ind, pop in enumerate(self.population):
            val = self.cost(pop)
            dict[ind] = val
            sum += val

        for j in dict.keys():
            dict[j] = dict[j]/sum
        res = sorted(dict.items(), key=lambda i: i[1])
        pop = []
        for i in range(int(self.pop_size/2)):
            p = self.selection(res)
            q = self.selection(res)
            r = np.random.rand()
            if r <= p_crossover:
                c1, c2 = self.crossover(p, q, self.crossover_type)
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
            else:
                c1, c2 = p, q
            pop.append(c1)
            pop.append(c2)

        self.population = pop

        return pop, res1

    def mutation(self, c):
        for i in range(self.tsp_len):
            r = np.random.rand()
            if r <= p_mutation:
                ind1 = i
                ind2 = np.random.randint(self.tsp_len)
                temp = c[ind1]
                c[ind1] = c[ind2]
                c[ind2] = temp
        return c

    def crossover(self, p, q, crossover_type):
        tsp_len = self.tsp_len
        while 1:
            cpoint_1 = np.random.randint(0, tsp_len)
            cpoint_2 = np.random.randint(0, tsp_len)
            if cpoint_1 == cpoint_2:
                continue
            else:
                if cpoint_1 > cpoint_2:
                    temp = cpoint_1
                    cpoint_1 = cpoint_2
                    cpoint_2 = temp
                break

        def crossover_PMX(p, q):
            child1 = copy.deepcopy(p)
            child2 = copy.deepcopy(q)
            child1[cpoint_1:cpoint_2+1] = q[cpoint_1:cpoint_2+1]
            child2[cpoint_1:cpoint_2+1] = p[cpoint_1:cpoint_2+1]
            child1_indices = [-1 for i in range(tsp_len)]
            for i in range(cpoint_1, cpoint_2+1):
                child1_indices[q[i]] = i
            for i in range(tsp_len):
                ind = child1[i]
                if i >= cpoint_1 and i <= cpoint_2:
                    continue
                while child1_indices[ind] != -1:
                    ind = child1_indices[ind]
                    ind = p[ind]

                child1[i] = ind

                child1_indices[ind] = i
            child2_indices = [-1 for i in range(tsp_len)]
            for i in range(cpoint_1, cpoint_2+1):
                child2_indices[p[i]] = i
            for i in range(tsp_len):
                ind = child2[i]
                if i >= cpoint_1 and i <= cpoint_2:
                    continue
                while child2_indices[ind] != -1:
                    ind = child2_indices[ind]
                    ind = q[ind]

                child2[i] = ind
                child2_indices[ind] = i
            return child1, child2

        def crossover_Cycle(p, q):
            p_indices = [-1 for i in range(tsp_len)]
            for i in range(tsp_len):
                p_indices[p[i]-1] = i
            q_indices = [-1 for i in range(tsp_len)]
            for i in range(tsp_len):
                q_indices[q[i]-1] = i
            c2 = [-1 for i in range(tsp_len)]
            fl = 1
            for i in range(0, tsp_len):
                t = i
                v = p[t]
                w = q[t]
                if c2[t] != -1:
                    continue
                if fl == 1:
                    while w not in c2:
                        c2[t] = w
                        t = p_indices[w-1]
                        v = p[t]
                        w = q[t]
                else:
                    while v not in c2:
                        c2[t] = v
                        t = q_indices[v-1]
                        v = p[t]
                        w = q[t]
                fl = not fl
            c1 = [-1 for i in range(tsp_len)]
            fl = 0
            for i in range(0, tsp_len):
                t = i
                v = p[t]
                w = q[t]
                if c1[t] != -1:
                    continue
                if fl == 1:
                    while w not in c1:
                        c1[t] = w
                        t = p_indices[w-1]
                        v = p[t]
                        w = q[t]
                else:
                    while v not in c1:
                        c1[t] = v
                        t = q_indices[v-1]
                        v = p[t]
                        w = q[t]
                fl = not fl
            return c1, c2

        def crossover_Order1(p, q):
            c1 = [-1 for i in range(tsp_len)]
            c1[cpoint_1:cpoint_2+1] = p[cpoint_1:cpoint_2+1]
            st_point = cpoint_1+1
            for i in range(tsp_len):
                if(c1[i] == -1):
                    while q[st_point] in c1:
                        st_point += 1
                        if(st_point == tsp_len):
                            st_point = 0
                    c1[i] = q[st_point]
            c1 = [-1 for i in range(tsp_len)]
            c1[cpoint_1:cpoint_2+1] = p[cpoint_1:cpoint_2+1]
            st_point = cpoint_1+1
            for i in range(tsp_len):
                if(c1[i] == -1):
                    while q[st_point] in c1:
                        st_point += 1
                        if(st_point == tsp_len):
                            st_point = 0
                    c1[i] = q[st_point]
            c2 = [-1 for i in range(tsp_len)]
            c2[cpoint_1:cpoint_2+1] = q[cpoint_1:cpoint_2+1]
            st_point = cpoint_1+1
            for i in range(tsp_len):
                if(c2[i] == -1):
                    while p[st_point] in c2:
                        st_point += 1
                        if(st_point == tsp_len):
                            st_point = 0
                    c2[i] = p[st_point]
            return c1, c2

        def crossover_Order2(p, q):
            inds = np.random.randint(tsp_len)
            while inds == 0:
                inds = np.random.randint(tsp_len)
            ind = []
            for i in range(inds):
                temp = np.random.randint(tsp_len)
                while temp in ind:
                    temp = np.random.randint(tsp_len)
                ind.append(temp)
            c1 = copy.deepcopy(p)
            c2 = copy.deepcopy(q)
            permute_cities = [q[i] for i in ind]
            for i in range(tsp_len):
                if(c1[i] in permute_cities):
                    c1[i] = -1
            c1
            k = 0
            for i in range(tsp_len):
                if c1[i] == -1:
                    c1[i] = permute_cities[k]
                    k += 1
            permute_cities = [p[i] for i in ind]
            for i in range(tsp_len):
                if(c2[i] in permute_cities):
                    c2[i] = -1
            k = 0
            for i in range(tsp_len):
                if c2[i] == -1:
                    c2[i] = permute_cities[k]
                    k += 1
            return c1, c2

        def crossover_Position(p, q):
            inds = np.random.randint(tsp_len)
            while inds == 0:
                inds = np.random.randint(tsp_len)
            ind = []
            for i in range(inds):
                temp = np.random.randint(tsp_len)
                while temp in ind:
                    temp = np.random.randint(tsp_len)
                ind.append(temp)
            c1 = copy.deepcopy(p)
            c2 = copy.deepcopy(q)
            for i in range(tsp_len):
                if i in ind:
                    c1[i] = q[i]
                else:
                    c1[i] = -1
            k = 0
            for i in range(tsp_len):
                if c1[i] == -1:
                    while k < tsp_len and p[k] in c1:
                        k += 1
                    c1[i] = p[k]
            for i in range(tsp_len):
                if i in ind:
                    c2[i] = p[i]
                else:
                    c2[i] = -1
            k = 0
            for i in range(tsp_len):
                if c2[i] == -1:
                    while k < tsp_len and q[k] in c2:
                        k += 1
                    c2[i] = q[k]
            return c1, c2

        def crossover_Genetic(p, q):
            d = {}
            for i in range(tsp_len-1):
                if p[i] not in d:
                    d[p[i]] = [p[i+1]]
                else:
                    d[p[i]].append(p[i+1])

            for i in range(tsp_len-1, 0, -1):
                if p[i] not in d:
                    d[p[i]] = [p[i-1]]
                else:
                    if p[i-1] not in d[p[i]]:
                        d[p[i]].append(p[i-1])

            if p[0] not in d[p[-1]]:
                d[p[-1]].append(p[0])
            if p[-1] not in d[p[0]]:
                d[p[0]].append(p[-1])

            for i in range(tsp_len-1):
                if q[i] not in d:
                    d[q[i]] = [q[i+1]]
                else:
                    if q[i+1] not in d[q[i]]:
                        d[q[i]].append(q[i+1])

            for i in range(tsp_len-1, 0, -1):
                if q[i] not in d:
                    d[q[i]] = [q[i-1]]
                else:
                    if q[i-1] not in d[q[i]]:
                        d[q[i]].append(q[i-1])

            if q[0] not in d[q[-1]]:
                d[q[-1]].append(q[0])
            if q[-1] not in d[q[0]]:
                d[q[0]].append(q[-1])
            c1 = []
            ind = 0
            large_ind = np.random.randint(tsp_len)
            e = copy.deepcopy(d)
            while 1:
                res = sorted(d.items(), key=lambda i: -len(i[1]))
                if len(res) == 0:
                    break
                if ind == 1:
                    large_ind = res[0][0]
                ind = 1
                c1.append(large_ind)
                del d[large_ind]
                for i in d.keys():
                    if large_ind in d[i]:
                        d[i].remove(large_ind)
            c2 = []
            ind = 0
            large_ind = np.random.randint(tsp_len)
            while 1:
                res = sorted(e.items(), key=lambda i: -len(i[1]))
                if len(res) == 0:
                    break
                if ind == 1:
                    large_ind = res[0][0]
                ind = 1
                c2.append(large_ind)
                del e[large_ind]
                for i in e.keys():
                    if large_ind in e[i]:
                        e[i].remove(large_ind)
            return c1, c2

        def crossover_MPX(p, q):
            c1 = [-1 for i in range(tsp_len)]
            k = 0
            for i in range(cpoint_1, cpoint_2+1):
                c1[k] = p[i]
                k += 1
            starting = cpoint_2-cpoint_1+1
            for i in range(tsp_len):
                if q[i] not in c1:
                    c1[starting] = q[i]
                    starting += 1
            c2 = [-1 for i in range(tsp_len)]
            k = 0
            for i in range(cpoint_1, cpoint_2+1):
                c2[k] = q[i]
                k += 1
            starting = cpoint_2-cpoint_1+1
            for i in range(tsp_len):
                if p[i] not in c2:
                    c2[starting] = p[i]
                    starting += 1
            return c1, c2

        def crossover_Alternation(p, q):
            c1 = [-1 for i in range(tsp_len)]
            k = 0
            for i in range(tsp_len):
                if k == tsp_len:
                    break
                if p[i] not in c1:
                    c1[k] = p[i]
                    k += 1
                if q[i] not in c1:
                    c1[k] = q[i]
                    k += 1
            c2 = [-1 for i in range(tsp_len)]
            k = 0
            for i in range(tsp_len):
                if k == tsp_len:
                    break
                if q[i] not in c2:
                    c2[k] = q[i]
                    k += 1
                if p[i] not in c2:
                    c2[k] = p[i]
                    k += 1
            return c1, c2
        if crossover_type == "PMX":
            c1, c2 = crossover_PMX(p, q)
        elif crossover_type == "Cycle":
            c1, c2 = crossover_Cycle(p, q)
        elif crossover_type == "Order1":
            c1, c2 = crossover_Order1(p, q)
        elif crossover_type == "Order2":
            c1, c2 = crossover_Order2(p, q)
        elif crossover_type == "Position":
            c1, c2 = crossover_Position(p, q)
        elif crossover_type == "Genetic":
            c1, c2 = crossover_Genetic(p, q)
        elif crossover_type == 'MPX':
            c1, c2 = crossover_MPX(p, q)
        elif crossover_type == "Alternation":
            c1, c2 = crossover_Alternation(p, q)
        else:
            print("Wrong choice")
            print(
                "Choose from 'PMX','Cycle','Order1','Order2','Position','Genetic','MPX','Alternation' ")
            exit()
        return c1, c2

    def run_algo(self):
        start = time.time()

        for i in range(self.iterations):
            dict = {}
            prev_pop = self.population
            pop1, res = self.pop_selection()
            res = res[:5]
            sum = 0.0
            for ind, pop in enumerate(self.population):
                val = 1/self.cost(pop)
                dict[ind] = val
            res2 = sorted(dict.items(), key=lambda i: i[1])

            # print(dict)

            j = 0
            # if elitism true 10% 5 out of 50 are compared of the previous population
            if self.elitism:
                for ind in range(len(res2)):
                    if ind == 5:
                        break
                    if res2[-ind][1] > res[j][1]:
                        self.population[res2[-ind][0]] = prev_pop[res[j][0]]
                        j += 1

            '''
            print("Genetation: {}".format(i),
                  "-- Population Size: {}".format(len(prev_pop)),
                  "-- BestFitness: {}".format((min(dict.values()))))
            '''

        end = time.time()
        total_time = round(end-start, 1)
        print("Total time: {}s".format(total_time))

        plt.plot(range(0, res2), res2, c='blue')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Function')
        plt.show()

        print("---------------------")
        return min(dict.values()), self.population[min(dict.items(), key=operator.itemgetter(1))[0]]


# Running experiments
if __name__ == "__main__":
    File.configsetters()

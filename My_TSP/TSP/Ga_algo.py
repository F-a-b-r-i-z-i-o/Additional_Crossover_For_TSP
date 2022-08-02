import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import operator

# Fixing the crossover and mutation probabilities

p_crossover = 0.95
p_mutation = 0.01

# Class Genetic Algo


class GAalgo:

    '''
        Inizialize constructor and generate randomly sequence of element
    '''

    def __init__(self, tsp_len, pop_size, weights, iterations, elitism, crossover_type, best_n):

        self.best_n = best_n

        self.all_fitness = []

        self.generation = []

        # Inizialize elitism

        self.elitism = elitism

        # Inizialize iteration

        self.iterations = iterations

        # Inizialize tsp_len

        self.tsp_len = tsp_len

        # Inizialize population size

        self.pop_size = pop_size

        # Inizialize type of crossover

        self.crossover_type = crossover_type

        # Enroll element in tsp

        elements = [i for i in range(tsp_len)]

        population = []

        # Select tsp_len randomly permutation return a permuted range.

        p = np.random.permutation(tsp_len)

        # Append element in population randomly

        for i in range(pop_size):
            population.append(list(np.array(elements)[p.astype(int)]))
            p = np.random.permutation(tsp_len)
        self.population = population
        self.weights = weights

    '''
        Calcolate the cost of solution
    '''

    def cost(self, sol):

        # Inizialize value
        value = 0.0

        # Inizialize weights

        weights = self.weights

        # Calcolate the cost of value with the weights

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
        elif crossover_type == 'MPX':
            c1, c2 = crossover_MPX(p, q)
        elif crossover_type == "Alternation":
            c1, c2 = crossover_Alternation(p, q)
        else:
            print("Wrong choice")
            print(
                "Choose from 'PMX','Cycle','Order1','Order2','Position','MPX','Alternation' ")
            exit()
        return c1, c2

    """
        Plot graph of best fitness
        by the iteration
    """

    def graph(self):
        plt.plot(self.generation, self.all_fitness,  c='blue')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Function')
        plt.show()

    def run_algo(self):
        start = time.time()
        self.all_fitness = []
        self.generation = []
        prova = []

        for i in range(self.best_n):
            prova.append(self.pop_selection())
            prova.sort()

            # print(len(prova))

        for i in range(self.iterations):
            self.generation.append(i)
            dict = {}
            prev_pop = self.population
            pop1, res = self.pop_selection()
            res = res[:5]
            sum = 0.0
            for ind, pop in enumerate(self.population):
                val = 1/self.cost(pop)
                dict[ind] = val
            res2 = sorted(dict.items(), key=lambda i: i[1])

            j = 0
            # if elitism true 10% 5 out of 50 are compared of the previous population
            if self.elitism:
                for ind in range(len(res2)):
                    if ind == 5:
                        break
                    if res2[-ind][1] > res[j][1]:
                        self.population[res2[-ind][0]] = prev_pop[res[j][0]]
                        j += 1

            values_min = min(dict.values())
            self.all_fitness.append(values_min)

            newA = res2[:self.best_n]

            newA.reverse()

            prova.extend(newA)

        '''
            if gen_number % 10 == 0:
                print(gen_number, values)

        print("\n----------------------------------------------------------------")
        print("Generation: " + str(gen_number))
        print("Fittest chromosome distance before training: " +
              str(values))
        print("Fittest chromosome distance after training: " +
              str(self.all_fitness[0]))
        # print("Target distance: " + str(TARGET))
        print("----------------------------------------------------------------\n")
        '''

        '''
            for v in res2[:self.best_n]:
                prova.append(v[0])
            '''
        '''
            print("Genetation: {}".format(i),
                  "-- Population Size: {}".format(len(prova)),
                  "-- BestFitness: {}".format((min(dict.values()))))
            '''
        '''
            for i in self.population:
                prova.append(i)
            '''

        end = time.time()
        total_time = round(end-start, 1)
        print("Total time: {}s".format(total_time))

        print("---------------------")
        return min(dict.values()), self.population[min(dict.items(), key=operator.itemgetter(1))[0]]

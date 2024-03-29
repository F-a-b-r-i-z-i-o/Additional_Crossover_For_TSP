###################################################
# Libraries used are:
# numpy -  To generate random integers as indices
# random - To shuffle the list of integers
# copy - To deepcopy a list
# time - To generate seeds for random numbers generator
###################################################


import matplotlib.pyplot as plt
import numpy as np
import copy
import time

# Fixing the crossover and mutation probabilities
p_crossover = 0.95
p_mutation = 0.01


# Class Genetic Algo
class GAalgo:

    '''
        Inizialize constructor and generate randomly sequence of element
    '''

    def __init__(self, tsp_len, weights, iterations, crossover_type):

        # List of all fitness
        self.all_fitness = []

        # List of generation
        self.generation = []

        # Inizialize iteration
        self.iterations = iterations

        # Inizialize tsp_len
        self.tsp_len = tsp_len

        # Inizialize type of crossover
        self.crossover_type = crossover_type

        # Enroll element in tsp
        elements = [i for i in range(tsp_len)]

        # List of population
        population = []

        # List of improvements
        self.improvements = []

        # List of cost parent and child
        self.l_cost = []

        # List of index parent and child
        self.index = []

        self.best_f = 1e300

        # Select tsp_len randomly permutation return a permuted range.
        p = np.random.permutation(tsp_len)

        # Swap and creare pop_size
        pop_size = self.tsp_len

        # Append element in population randomly
        for i in range(pop_size):
            population.append(list(np.array(elements)[p.astype(int)]))
            p = np.random.permutation(tsp_len)

        self.population = population
        self.weights = weights

    '''
       Represents the objective function
       And
       Calcolate the cost of solution
    '''

    def cost(self, sol):

        # Inizialize value
        value = 0.0

        # Inizialize weights
        weights = self.weights

        # Calcolate the cost of path with the weights
        for i in range(self.tsp_len-1):

            # Value cost
            value += weights[sol[i-1]][sol[i]]

        value += weights[sol[-1]][sol[0]]

        # Return value normalize
        return 1/value

    '''
        Function to select pop randomly
    '''

    def selection(self, res):

        # Inizialize sum
        sum = 0.0

        # Create a matrix of the given shape and populate it with random samples from a uniform distribution on [0,1]
        rvalue = np.random.rand()

        # Create pop after comparison with rvalue
        j = 0
        for i in res:
            sum += i[1]
            if(sum >= rvalue):
                ret = j
                break
            j += 1
        return self.population[res[j][0]]

    '''
        Roulette Wheel
        Select an individual randomly according to the
        (proportional) probability of fitness F for each individual.
        The optimization problem is a minimization problem:
        F must be a decreasing transformation f(x)=1/f(x).
    '''

    def roulette_wheel(self):

        # Create dict to append result
        dict = {}

        # Inizialize sum
        sum = 0.0

        for ind, pop in enumerate(self.population):

            # Calcolate cost of pop
            val = self.cost(pop)

            # Normalize data and save dict
            dict[ind] = 1/val

        # Save result order min to max
        res1 = sorted(dict.items(), key=lambda i: i[1])

        # Create other dict to append result
        dict = {}

        for ind, pop in enumerate(self.population):

            # Calcolate the cost of pop
            val = self.cost(pop)

            # Save cost in dict
            dict[ind] = val

            # Update the sum
            sum += val

        for j in dict.keys():

            # Divide key for sum
            dict[j] = dict[j]/sum

        # Order result save in a dict min to max
        res = sorted(dict.items(), key=lambda i: i[1])

        # List of pop
        pop = []

        # List of parents
        parents = []

        # Swap pop_size
        pop_size = self.tsp_len

        # Select copule of pop randomly
        for i in range(int(pop_size/2)):
            p1 = self.selection(res)
            p2 = self.selection(res)
            r = np.random.rand()

            # control random value is better than p_crossover
            if r <= p_crossover:

                # execute crossover and mutation
                c1, c2 = self.crossover(p1, p2, self.crossover_type)

                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
            else:

                c1, c2 = p1, p2

            # Append child1, child2 in new pop
            pop.append(c1)
            pop.append(c2)

            # Append parents in parents list
            parents.append(p1)
            parents.append(p2)

        self.population = pop

        # Return pop value of child, parents and value of parents
        return pop, res1, parents, res

    """
        Classic Mutation
        Mutation means altering the chromosome of the children.
        children can be either copies of the parents or produced by crossover.
        Can be used when chromosomes are vectors or strings.
        Alters each gene with a probability p_mutation
    """

    def mutation(self, c):

        for i in range(self.tsp_len):

            # Create a matrix of the given shape and populate it with random samples from a uniform distribution on [0,1]
            r = np.random.rand()

            # Control il r value matrix is small than p_mutation
            if r <= p_mutation:

                # First index
                ind1 = i

                # Second index is randomly in tsp_len
                ind2 = np.random.randint(self.tsp_len)

                # Swap the index of child
                temp = c[ind1]
                c[ind1] = c[ind2]
                c[ind2] = temp

        return c

    '''
        Wrapper per crossover
        Richiamo crossover selezionato
    '''

    def crossover(self, p1, p2, crossover_type):

        # Inizialize tsp_len
        tsp_len = self.tsp_len

        # Create 2 cat point random to [0, tsp_len]
        while 1:
            cpoint_1 = np.random.randint(0, tsp_len)
            cpoint_2 = np.random.randint(0, tsp_len)

            # Control equality cut point
            if cpoint_1 == cpoint_2:
                continue

            # If not equal swap cpoint
            else:
                if cpoint_1 > cpoint_2:
                    temp = cpoint_1
                    cpoint_1 = cpoint_2
                    cpoint_2 = temp
                break

        # Crossover PMX

        def crossover_PMX(p1, p2):
            '''
                L'operatore di crossover parzialmente mappato è stato proposto
                da Gold- berg e Lingle (1985). Esso trasmette le informazioni sull'ordine e
                sul valore dai tour dei genitori ai tour della progenie. Una parte della stringa
                di un genitore viene mappata su una parte della stringa dell'altro genitore e
                le informazioni rimanenti vengono scambiate. Si considerino, ad esempio, i
                seguenti due tour di genitori:
                (1 2 3 4 5 6 7 8) e
                (3 7 5 1 6 8 2 4))
                L'operatore PMX crea una progenie nel modo seguente. Innanzitutto,
                seleziona in modo uniforme e casuale due punti di taglio lungo le stringhe,
                che rappresentano i tour dei genitori. Supponiamo che il primo punto di
                taglio sia selezionato tra il terzo e il quarto elemento della stringa e il
                secondo tra il sesto e il settimo elemento della stringa. Ad esempio,
                (1 2 3j4 5 6j7 8) e
                (3 7 5j1 6 8j2 4))
                Le sottostringhe tra i punti di taglio sono chiamate sezioni di mappatura.
                Nel nostro esempio, esse definiscono le mappature 4 +- 1, 5 +- 6 e 6 +- 8.
                Ora la sezione di mappatura del primo genitore viene copiata nella seconda
                discendenza e la sezione di mappatura del secondo genitore viene copiata
                nella prima discendenza, crescendo:
                prole 1: (x xj1 6 8jx x) e
                prole 2: (x x xj4 5 6jx x))
                Quindi la progenie i (i = 1,2) viene riempita copiando gli elementi del
                genitore i-esimo. Nel caso in cui una città sia già presente nella progenie,
                viene sostituita in base alle mappature.
                Ad esempio, il primo elemento della progenie 1 sarà un 1
                come il primo elemento del primo genitore. Tuttavia, nella progenie 1 è già
                presente un 1. Quindi, a causa della mappatura 1 +- 4, scegliamo che il
                primo elemento della progenie 1 sia un 4. Il secondo, il terzo e il settimo
                elemento della progenie 1 possono essere presi dal primo genitore. Tuttavia,
                l'ultimo elemento della progenie 1 sarebbe un 8, che è già presente.
                A causa delle mappature 8 +- 6 e 6 +- 5, si sceglie che sia un 5. Quindi,
                progenie 1: (4 2 3j1 6 8j7 5))
                Analogamente, troviamo
                progenie 2: (3 7 8j4 5 6j2 1))
                Si noti che le posizioni assolute di alcuni elementi di entrambi i genitori
                vengono conservate.
                Una variante dell'operatore PMX è descritta in Grefenstette (1987b): dati
                due genitori, la progenie viene creata come segue. Per prima cosa, la stringa
                del secondo genitore viene copiata nella progenie. Successivamente, si
                sceglie un subtour arbitrario dal primo genitore. Infine, si apportano alla
                discendenza le modifiche minime necessarie per ottenere il subtour scelto.
                Ad esempio, si considerino i tour dei genitori
                (1 2 3 4 5 6 7 8) e
                (1 5 3 7 2 4 6 8))
                e supponiamo che venga scelto il subtour (3 4 5). In questo modo si ottiene la
                progenie
                (1 3 4 5 7 2 6 8))
            '''

            # Create child1
            child1 = copy.deepcopy(p1)

            # Create child2
            child2 = copy.deepcopy(p2)

            # Select cat point by the child that rappresent the tour of parent
            child1[cpoint_1:cpoint_2+1] = p2[cpoint_1:cpoint_2+1]
            child2[cpoint_1:cpoint_2+1] = p1[cpoint_1:cpoint_2+1]

            # Inizialize indices of child1
            child1_indices = [-1 for i in range(tsp_len)]

            # Enroll catpoint1 to cpoint2
            for i in range(cpoint_1, cpoint_2+1):
                # Save new index of child1
                child1_indices[p2[i]] = i

            # Check that i is included in the cut points
            for i in range(tsp_len):
                ind = child1[i]
                if i >= cpoint_1 and i <= cpoint_2:
                    continue

                # Transfer index of child1 to parent q and child
                while child1_indices[ind] != -1:
                    ind = child1_indices[ind]
                    ind = p1[ind]

                child1[i] = ind

                child1_indices[ind] = i

            # Inizialize indices of child2
            child2_indices = [-1 for i in range(tsp_len)]

            # Enroll catpoint1 to cpoint2
            for i in range(cpoint_1, cpoint_2+1):
                # Save new index of childe2
                child2_indices[p1[i]] = i

            # Check that i is included in the cut points
            for i in range(tsp_len):
                ind = child2[i]

                if i >= cpoint_1 and i <= cpoint_2:
                    continue

                # Transfer index of child2 to parent q and child2
                while child2_indices[ind] != -1:
                    ind = child2_indices[ind]
                    ind = p2[ind]

                child2[i] = ind
                child2_indices[ind] = i

            return child1, child2

        # Crossover Cycle

        def crossover_Cycle(p1, p2):
            '''
                L'operatore di crossover ciclico è stato proposto da Oliver et al.
                (1987). Cerca di creare una progenie dai genitori in cui ogni posizione è
                occupata da un elemento corrispondente di uno dei genitori. Ad esempio, si
                considerino nuovamente i genitori
                (1 2 3 4 5 6 7 8) e
                (2 4 6 8 7 5 3 1))
                Ora scegliamo che il primo elemento della progenie sia il primo elemento
                del primo tour dei genitori o il primo elemento del secondo tour dei
                genitori. Quindi, il primo elemento della progenie deve essere un 1 o un 2.
                Supponiamo di sceglierlo come 1,
                (1 * * * * * * *))
                Consideriamo ora l'ultimo elemento della discendenza. Poiché questo
                elemento deve essere scelto da uno dei genitori, può essere solo un 8 o un
                1. Tuttavia, se si scegliesse un 1, la progenie non rappresenterebbe un giro
                legale. Pertanto, si sceglie un 8,
                (1 * * * * * * 8))
                Analogamente, troviamo che anche il quarto e il secondo elemento della
                progenie devono essere selezionati dal primo genitore, il che risulta in
                (1 2 * 4 * * * 8))
                Le posizioni degli elementi scelti finora sono dette un ciclo.
                Consideriamo ora il terzo elemento della progenie. Questo elemento può
                essere scelto da uno qualsiasi dei genitori. Supponiamo di sceglierlo dal
                genitore 2. Ciò implica che anche il quinto, il sesto e il settimo elemento
                della discendenza devono essere scelti dal secondo genitore, poiché
                formano un altro ciclo. Si ottiene quindi la seguente discendenza:
                (1 2 6 4 7 5 3 8))
                La posizione assoluta della metà degli elementi di entrambi i genitori
                viene conservata. Oliver et al. (1987) hanno concluso, sulla base di risultati
                teorici ed empirici, che l'operatore CX fornisce risultati migliori per il
                Travelling Salesman Problem rispetto all'operatore PMX.
            '''

            # Inizialize parent p indices
            p1_indices = [-1 for i in range(tsp_len)]

            # Take parent p indecs to tsp -1
            for i in range(tsp_len):
                p1_indices[p1[i]-1] = i

            # Inizialize parent q indices
            p2_indices = [-1 for i in range(tsp_len)]

            # Take parent q indecs to tsp -1
            for i in range(tsp_len):
                p2_indices[p2[i]-1] = i

            # Inizialize child2
            c2 = [-1 for i in range(tsp_len)]

            # First element
            fl = 1

            for i in range(0, tsp_len):

                # Index
                t = i

                # Parent p
                v = p1[t]

                # Parent q
                w = p2[t]

                # Control over position
                if c2[t] != -1:
                    continue

                # First postion
                if fl == 1:

                    # Check and adjust the values in w
                    while w not in c2:
                        c2[t] = w
                        t = p1_indices[w-1]
                        v = p1[t]
                        w = p2[t]
                else:

                    # Check and adjust the values in v
                    while v not in c2:
                        c2[t] = v
                        t = p2_indices[v-1]
                        v = p1[t]
                        w = p2[t]

                # Return not first
                fl = not fl

            # Inizialize child1
            c1 = [-1 for i in range(tsp_len)]

            # First element 0
            fl = 0

            for i in range(0, tsp_len):

                # Index
                t = i

                # Parent p
                v = p1[t]

                # Parent q
                w = p2[t]

                # Control over position
                if c1[t] != -1:
                    continue

                 # First postion
                if fl == 1:

                    # Check and adjust the values in w
                    while w not in c1:
                        c1[t] = w
                        t = p1_indices[w-1]
                        v = p1[t]
                        w = p2[t]
                else:

                    # Check and adjust the values in v
                    while v not in c1:
                        c1[t] = v
                        t = p2_indices[v-1]
                        v = p1[t]
                        w = p2[t]
                fl = not fl
            return c1, c2

        # Crossover Order1 (OX1)

        def crossover_Order1(p1, p2):
            '''
                L'operatore order crossover è stato proposto da Davis (1985).
                L'OX1 sfrutta una proprietà della rappresentazione dei percorsi, secondo
                cui l'ordine delle città (e non la loro posizione) è importante. Costruisce una
                progenie scegliendo una città
                di un sottotour di un genitore e preservando l'ordine relativo delle città
                dell'altro genitore. Ad esempio, si considerino i seguenti due tour di
                genitori:
                (1 2 3 4 5 6 7 8) e
                (2 4 6 8 7 5 3 1))
                e supponiamo di selezionare un primo punto di taglio tra il secondo e il
                terzo bit e un secondo tra il quinto e il sesto bit. Quindi,
                (1 2j3 4 5j6 7 8) e
                (2 4j6 8 7j5 3 1))
                La progenie viene creata nel modo seguente. In primo luogo, i segmenti
                del tour tra il punto di taglio vengono copiati nella progenie, il che dà come
                risultato
                (* *j3 4 5j* * *) e
                (* *j6 8 7j* * *))
                Quindi, a partire dal secondo punto di taglio di un genitore, si copiano le
                altre città nell'ordine in cui appaiono nell'altro genitore, sempre a partire dal
                secondo punto di taglio e omettendo le città già presenti. Quando si
                raggiunge la fine della stringa del genitore, si continua dalla sua prima
                posizione. Nel nostro esempio si ottengono i seguenti figli:
                (8 7j3 4 5j1 2 6) e
                (4 5j6 8 7j1 2 3))
            '''

            # Inizialize child1
            c1 = [-1 for i in range(tsp_len)]

            # Create 2 cat point child1
            c1[cpoint_1:cpoint_2+1] = p1[cpoint_1:cpoint_2+1]

            # Create start point chiild1
            st_point = cpoint_1+1

            for i in range(tsp_len):

                # Control out index
                if(c1[i] == -1):

                    while p2[st_point] in c1:

                        # update start point
                        st_point += 1

                        # Control the end
                        if(st_point == tsp_len):
                            st_point = 0

                    # Assign child 1 to new parent q start point
                    c1[i] = p2[st_point]

            # Inizialize child2
            c2 = [-1 for i in range(tsp_len)]

            # Create cat point child2
            c2[cpoint_1:cpoint_2+1] = p2[cpoint_1:cpoint_2+1]

            # Create start point child2
            st_point = cpoint_1+1

            for i in range(tsp_len):

                # Control out index
                if(c2[i] == -1):
                    while p1[st_point] in c2:

                        # update start point
                        st_point += 1

                        # Control the end
                        if(st_point == tsp_len):

                            st_point = 0

                     # Assign child2 to new parent p start point
                    c2[i] = p1[st_point]

            return c1, c2

        # Crossover Order2 (OX2)

        def crossover_Order2(p1, p2):
            '''
                L'operatore di crossover basato sull'ordine (Syswerda 1991) seleziona a
                caso diverse posizioni in un giro di genitori e l'ordine delle città nelle
                posizioni selezionate di questo genitore viene imposto all'altro genitore. Ad
                esempio, consideriamo nuovamente i genitori
                (1 2 3 4 5 6 7 8) e
                (2 4 6 8 7 5 3 1))
                e supponiamo che nel secondo genitore vengano selezionate la seconda, la
                terza e la sesta posizione. Le città presenti in queste posizioni sono
                rispettivamente città 4, città 6 e città 5. Nel primo genitore queste città sono
                presenti nelle posizioni quarta, quinta e sesta. Ora la progenie è uguale al
                genitore 1 tranne che per la quarta, quinta e sesta posizione:
                (1 2 3 * * * 7 8))
                Aggiungiamo le città mancanti alla progenie nello stesso ordine in cui
                appaiono nel secondo tour dei genitori. Il risultato è
                (1 2 3 4 6 5 7 8))
                Scambiando il ruolo del primo genitore e del secondo genitore si ottiene,
                utilizzando le stesse posizioni selezionate,
                (2 4 3 8 7 5 6 1))
            '''

            # Select random position
            inds = np.random.randint(tsp_len)

            while inds == 0:
                inds = np.random.randint(tsp_len)

            # List of index
            ind = []

            # Select random index
            for i in range(inds):
                temp = np.random.randint(tsp_len)
                while temp in ind:
                    temp = np.random.randint(tsp_len)
                ind.append(temp)

            # Copy p in child1
            c1 = copy.deepcopy(p1)

            # Copy p in childe2
            c2 = copy.deepcopy(p2)

            # Create permute cities by parent p1
            permute_cities = [p1[i] for i in ind]

            for i in range(tsp_len):

                # Control child1 in permute cities
                if(c1[i] in permute_cities):

                    # Decrement value
                    c1[i] = -1
            k = 0
            for i in range(tsp_len):

                # Control first element
                if c1[i] == -1:

                    # Assign child1 new permute cities
                    c1[i] = permute_cities[k]

                    # Incremnt k
                    k += 1

            # Create permute cities by parent p1
            permute_cities = [p1[i] for i in ind]

            for i in range(tsp_len):

                # Control child2 in permute cities
                if(c2[i] in permute_cities):

                    # Decrement value
                    c2[i] = -1

            k = 0
            for i in range(tsp_len):

                # Control first element
                if c2[i] == -1:

                    # Assign child2 new permute cities
                    c2[i] = permute_cities[k]

                    # Increment k
                    k += 1

            return c1, c2

        # Crossover Position
        def crossover_Position(p1, p2):
            '''
                Anche l'operatore basato sulla posizione (Syswerda 1991) inizia
                selezionando un insieme casuale di posizioni nei tour dei genitori. Tuttavia,
                questo operatore impone la posizione delle città selezionate alle città
                corrispondenti dell'altro genitore. Ad esempio, si considerino i tour dei
                genitori
                (1 2 3 4 5 6 7 8) e
                (2 4 6 8 7 5 3 1))
                e supponiamo che vengano selezionate la seconda, la terza e la sesta
                posizione. Questo porta alla seguente progenie:
                (1 4 6 2 3 5 7 8) e
                (4 2 3 8 7 6 5 1))
            '''

            # Select random index
            inds = np.random.randint(tsp_len)

            while inds == 0:

                inds = np.random.randint(tsp_len)

            # Create list of index
            ind = []

            # Enroll all inds create
            for i in range(inds):

                # Append all temp i ind
                temp = np.random.randint(tsp_len)

                while temp in ind:

                    temp = np.random.randint(tsp_len)

                ind.append(temp)

            # Copy p in child1
            c1 = copy.deepcopy(p1)

            # Copy q in child2
            c2 = copy.deepcopy(p2)

            for i in range(tsp_len):

                # Control all index select in child1 are the same of city parent q
                if i in ind:
                    c1[i] = p1[i]
                else:
                    c1[i] = -1

            k = 0
            for i in range(tsp_len):

                # Control all index select in child1  are the same of city parent p
                if c1[i] == -1:
                    while k < tsp_len and p1[k] in c1:
                        k += 1
                    c1[i] = p1[k]

            for i in range(tsp_len):

                # Control all index select in child1 are the same of city parent p
                if i in ind:
                    c2[i] = p1[i]
                else:
                    c2[i] = -1

            k = 0
            for i in range(tsp_len):

                # Control all index select in child1 are the same of city parent q
                if c2[i] == -1:
                    while k < tsp_len and p2[k] in c2:
                        k += 1
                    c2[i] = p2[k]
            return c1, c2

        # Crossover a posizione alternata (AP)
        def crossover_Alternation(p1, p2):
            '''
                L'operatore di crossover a posizione alternata (Larranaga et al. 1996a)
                crea semplicemente una progenie selezionando alternativamente l'elemento
                successivo del primo genitore e l'elemento successivo del secondo genitore,
                omettendo gli elementi già presenti nella progenie. Ad esempio, se il
                genitore 1 è
                (1 2 3 4 5 6 7 8)
                e il genitore 2 è
                (3 7 5 1 6 8 2 4))
                l'operatore AP dà la seguente discendenza
                (1 3 2 7 5 4 6 8))
                Scambiando i genitori si ottiene
                (3 1 7 2 5 4 6 8))
            '''

            # Inizialize child1
            c1 = [-1 for i in range(tsp_len)]

            k = 0
            for i in range(tsp_len):

                # Control k enroll all element
                if k == tsp_len:
                    break

                # Control the element of parent p1 not in in childern1
                if p1[i] not in c1:
                    c1[k] = p1[i]
                    k += 1

                # Control the element of parent p2 not in in childern1
                if p2[i] not in c1:
                    c1[k] = p2[i]
                    k += 1

            # Inizialize child2
            c2 = [-1 for i in range(tsp_len)]

            k = 0
            for i in range(tsp_len):

                # Control k enroll all element
                if k == tsp_len:
                    break

                # Control the element of parent p2 not in in childern2
                if p2[i] not in c2:
                    c2[k] = p2[i]
                    k += 1

                # Control the element of parent p1 not in in childern2
                if p1[i] not in c2:
                    c2[k] = p1[i]
                    k += 1
            return c1, c2

        if crossover_type == "PMX":
            c1, c2 = crossover_PMX(p1, p2)
        elif crossover_type == "Cycle":
            c1, c2 = crossover_Cycle(p1, p2)
        elif crossover_type == "Order1":
            c1, c2 = crossover_Order1(p1, p2)
        elif crossover_type == "Order2":
            c1, c2 = crossover_Order2(p1, p2)
        elif crossover_type == "Position":
            c1, c2 = crossover_Position(p1, p2)
        elif crossover_type == "Alternation":
            c1, c2 = crossover_Alternation(p1, p2)
        else:
            print("Bad Choice")
            print(
                "Choose from 'PMX','Cycle','Order1','Order2','Position','Alternation'")
            exit()
        return c1, c2

    """
        Plot graph of best fitness
        by the iteration
    """

    def graph(self):

        # Plot graph where on y: all_fitness x: number of generation

        plt.plot(self.generation, self.all_fitness,  c='blue')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Function')
        plt.show()

    '''
        Function that update the best value
    '''

    def update_best(self, x, fx):

        fx = fx[1]

        if fx < self.best_f:

            self.best_f = fx
            self.best = x
            self.improvements.append((fx))

    """
    
    The run_algo function: 
    
        1. Put the fathers (variable l) and children together.
        2. Calculated the objective function for each child (res2).
        3. Put together, in a single list, the objective functions of the fathers (which I already had) and the children. (self.l_cost)
        4. Created a list that contains the indexes of all (both fathers and children)
        5. Sorted this list of indexes by the value of the function f. (self.l1)
        6. Took the first part of these indexes. (l1best)
        7. Reconstructed the new population by taking the values of l only for better indices and the values of f.
        Finally, I called the update_best function.
        
    """

    def run_algo(self):

        pop_size = self.tsp_len

        # Start run algo in seconds
        start = time.time()

        for i in range(1, self.iterations+1):

            # Append generation
            self.generation.append(i)

            # Extract childe and parent
            child, res, parents, res2 = self.roulette_wheel()

            # Append index parents
            self.index.append(parents)

            # Append index child
            self.index.append(child)

            # Append cost child
            self.l_cost.append(res)

            # Append cost parents
            self.l_cost.append(res2)

            # Crate variable with parents and child index
            l = parents + child

            # Create variable with all cost
            f = res + res2

            # Create list with all_pop
            l1 = list(range(2*pop_size))

            # order list of index in base at value o function f
            l1.sort(key=lambda i: f[i])

            # Take first part of list
            l1best = l1[:pop_size]

            # Recostruction new population with l value only for best index and value of f
            po = [l[i] for i in l1best]
            f_obj = [f[i] for i in l1best]

            # Recall function update best
            self.update_best(po[0], f_obj[1])

            path = po[0]

            # Save the best min values
            best_values = min(self.improvements)

            # Approximation best_values
            best_values = round(best_values)

            # Save best on all_best_fitness
            self.all_fitness.append(best_values)

            print("Genetation: {}".format(i),
                  "-- BestFitness: {}".format(best_values))

            # Stop the time of algoritm
            end = time.time()

            # Calculate time of execution for best solution
            total_time = round(end-start, 1)

            # Control total time is lower than 5 minutes
            if total_time > 500:
                break

        print("---------------------")

        print("Total time: {}s".format(total_time))

        print("---------------------")

        print("BEST SOLUTION: {}".format(best_values))

        print("---------------------")

        # Return the best path and the best cost of GA_Algo
        return (best_values, path)

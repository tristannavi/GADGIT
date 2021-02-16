import random
import sys

import numpy as np
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# random.seed(64)

data_frame = pd.read_pickle(sys.argv[1])

MAX_ITEM = 100
NBR_ITEMS = data_frame.shape[0]

global_frontier = [0 for x in range(NBR_ITEMS)]

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", set, fitness=creator.Fitness)

def dummy_init(val):
    return 100

toolbox = base.Toolbox()
# toolbox.register("attr_item", random.randrange, NBR_ITEMS)
# toolbox.register("attr_item", dummy_init, NBR_ITEMS)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, MAX_ITEM)
toolbox.register("indices", random.sample, range(NBR_ITEMS), MAX_ITEM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    if len(individual) != MAX_ITEM:
        raise ValueError('Indiv has wrong number of things')

    between = 0.0
    for item in individual:
        # degree += data_frame.loc[item, 'Degree']
        # between += data_frame.loc[item, 'Betweenness']
        # degree += data_frame.iloc[item]['Degree']
        between += data_frame.iloc[item]['Betweenness']
        global_frontier[item] += 1
    if len(individual) > MAX_ITEM:
        return -10000, # -10000
    # return degree, between
    return between, 
    
def cxSDB(ind1, ind2):
    """SDB Crossover as defined in thesis."""
    intersect = ind1.intersection(ind2) # New indivs start with intersection

    # Grab the other genes and shuffle them
    dealer = list(ind1.union(ind2) - intersect)
    random.shuffle(dealer)

    if len(dealer) % 2 != 0:
        raise ValueError('Dealer assumption on indiv crossover failure')

    # Clear old sets
    ind1.clear()
    ind2.clear()

    # Split list in half and assign to new indivs
    ind1.update(dealer[:len(dealer)//2])
    ind1.update(intersect)
    ind2.update(dealer[len(dealer)//2:])
    ind2.update(intersect)

    if len(ind1) != MAX_ITEM or len(ind2) != MAX_ITEM:
        raise ValueError('SDB created invalid indiv')
    
    return ind1, ind2

def mutFlipper(individual):
    if len(individual) == MAX_ITEM: # Not sure if empty is truly possible; keeping for now.
        off_index = random.choice(list(set(range(0, NBR_ITEMS)) - individual))
        individual.remove(random.choice(sorted(tuple(individual))))
        individual.add(off_index)
        if len(individual) != MAX_ITEM:
            raise ValueError('Mutation created an invalid indiv')
    else:
        raise ValueError('Mutation received invalid indiv')

    return individual,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSDB)
toolbox.register("mutate", mutFlipper)

def main():
    # random.seed(64)
    NGEN = 100
    NPOP = 25
    CXPB = 0.75
    MUTPB = 0.25
    NK = 3
    
    toolbox.register("select", tools.selTournament, tournsize=NK)
    pop = toolbox.population(n=NPOP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats, halloffame=hof)
    
    return pop, stats, hof
                 
if __name__ == "__main__":
    pop, stats, hof = main()
    elite = hof[0]
    print('Size: ', len(elite))
    buf = 'Index in elite: '
    for ind in sorted(list(elite)):
        buf += str(ind) + ', '
    print(buf)
    buf = 'Genes in elite: '
    for ind in sorted(list(elite)):
        buf += data_frame.iloc[ind]['GeneName'] + ', '
    print(buf)
    print('Nodes exloration count: ')
    print(global_frontier)
    missed_nodes = [data_frame.iloc[ind]['GeneName'] for ind, x in enumerate(global_frontier) if x == 0]
    print('Nodes never explored (bad): N =', len(missed_nodes))
    print(', '.join(missed_nodes))

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

IND_INIT_SIZE = 100
MAX_ITEM = 100
NBR_ITEMS = data_frame.shape[0]

global_frontier = [0 for x in range(NBR_ITEMS)]

creator.create("Fitness", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.randrange, NBR_ITEMS)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    degree = 0.0
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

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2
    
def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

def mutFlipper(individual):
    """Mutation as specified in thesis."""
    if len(individual) > 0: # Not sure if empty is truly possible; keeping for now.
        off_index = random.choice(list(set(range(0, NBR_ITEMS)) - individual))
        individual.remove(random.choice(sorted(tuple(individual))))
        individual.add(off_index)
    else: # Mutation rebuilds if empty.
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", cxSet)
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

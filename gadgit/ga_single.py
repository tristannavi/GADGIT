import random
import sys

import numpy as np
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from GAInfo import GAInfo

FRONTIER = [0 for x in range(GENE_COUNT)]

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(GENE_COUNT), COM_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def single_eval(individual):
    assert len(individual) == COM_SIZE, 'Indiv does not match community size in eval'

    fit_sum = 0.0
    for item in individual:
        fit_sum += data_frame.iloc[item]['Betweenness']
        FRONTIER[item] += 1
    
    return fit_sum, 
    
def cxSDB(ind1, ind2):
    """SDB Crossover as defined in thesis."""
    intersect = ind1.intersection(ind2) # New indivs start with intersection

    # Grab the other genes and shuffle them
    dealer = list(ind1.union(ind2) - intersect)
    random.shuffle(dealer)

    assert len(dealer) % 2 == 0, 'Dealer assumption on indiv crossover failure'

    # Clear old sets
    ind1.clear()
    ind2.clear()

    # Split list in half and assign to new indivs
    ind1.update(dealer[:len(dealer)//2])
    ind1.update(intersect)
    ind2.update(dealer[len(dealer)//2:])
    ind2.update(intersect)

    assert len(ind1) == COM_SIZE and len(ind2) == COM_SIZE, 'SDB created invalid individual'
    
    return ind1, ind2

def mutFlipper(individual):
    assert len(individual) == COM_SIZE, 'Mutation received invalid indiv'
    off_index = random.choice(list(set(range(0, GENE_COUNT)) - individual))
    individual.remove(random.choice(sorted(tuple(individual))))
    individual.add(off_index)
    assert len(individual) == COM_SIZE, 'Mutation created an invalid indiv'

    return individual,

def ga_single(gene_info, ga_info):
    toolbox.register("evaluate", single_eval)
    toolbox.register("mate", cxSDB)
    toolbox.register("mutate", mutFlipper)
    toolbox.register("select", tools.selTournament, tournsize=ga_info.nk)
    pop = toolbox.population(n=ga_info.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaSimple(pop, toolbox, ga_info.cxpb, ga_info.mutpb, ga_info.gen, stats, halloffame=hof)
    
    return pop, stats, hof

if __name__ == "__main__":
    ga_info = GAInfo(generation=10)
    pop, stats, hof = ga_single(ga_info)
    post_run(None, ga_info, pop, stats, hof)

import random
import sys

import numpy as np
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from .GAInfo import GAInfo
from .GeneInfo import GeneInfo
from .post_run import post_run

def single_eval(gene_info, individual):
    """ Single objective summation of the centrality of a particular frame's chosen column.
    
    Due to gene_info.obj_list obviously accepting a list for the purposes of extending to MOP,
    in the case of this single_eval, the head of the list is treated as the 'single' objective.
    """

    assert len(individual) == gene_info.com_size, 'Indiv does not match community size in eval'
    assert set(gene_info.fixed_list_ids).issubset(individual), 'Indiv does not possess all fixed genes'

    fit_col = gene_info.obj_list[0]
    fit_sum = 0.0
    for item in individual:
        fit_sum += gene_info.data_frame.iloc[item][fit_col]
        gene_info.frontier[item] += 1
    
    return fit_sum, 
    
def cxSDB(gene_info, ind1, ind2):
    """SDB Crossover
    
    Computes the intersection and asserts that after the intersection,
    the amount of genes left over to 'deal' between two new individuals is even.

    Clears the set structures of their old information, updates with the intersection,
    and lastly hands out half of the shuffled dealer to each indiv.

    ind1 and ind2 and kept as objects since they inherit from set, but have additional properties.

    Note that this process is not destructive to the fixed genes inside of the individuals.
    """

    # Build dealer
    intersect = ind1.intersection(ind2)
    dealer = list(ind1.union(ind2) - intersect)
    random.shuffle(dealer)
    assert len(dealer) % 2 == 0, 'Dealer assumption on indiv crossover failure'

    # Rebuild individuals and play out dealer
    ind1.clear()
    ind2.clear()
    ind1.update(dealer[:len(dealer)//2])
    ind1.update(intersect)
    ind2.update(dealer[len(dealer)//2:])
    ind2.update(intersect)
    assert len(ind1) == gene_info.com_size and len(ind2) == gene_info.com_size, 'SDB created invalid individual'
    assert set(gene_info.fixed_list_ids).issubset(ind1), 'Ind1 does not possess all fixed genes after crossover'
    assert set(gene_info.fixed_list_ids).issubset(ind2), 'Ind2 does not possess all fixed genes after crossover'

    return ind1, ind2

def mutFlipper(gene_info, individual):
    """Flip based mutation. Flip one off to on, and one on to off.
    
    Must not allow the choice of a fixed gene to be turned off.
    """
    assert len(individual) == gene_info.com_size, 'Mutation received invalid indiv'
    off_index = random.choice(list(set(range(0, gene_info.gene_count)) - individual))
    individual.remove(random.choice(sorted(tuple(individual - set(gene_info.fixed_list_ids)))))
    individual.add(off_index)
    assert len(individual) == gene_info.com_size, 'Mutation created an invalid indiv'
    assert set(gene_info.fixed_list_ids).issubset(individual), 'Individual does not possess all fixed genes after mutation'

    return individual,

def indiv_builder(gene_info):
    """Implementation of forcing fixed genes in creation of new individual."""
    num_choices = gene_info.com_size - len(gene_info.fixed_list)
    valid_choices = list(set(range(gene_info.gene_count)) - set(gene_info.fixed_list_ids))
    base_indiv = random.sample(valid_choices, num_choices)
    base_indiv.extend(gene_info.fixed_list_ids)
    return base_indiv

def ga_single(gene_info, ga_info):
    """Main loop which sets DEAP objects and calls EA algorithm.
    
    Parameters
    -------
    gene_info, GeneInfo class
        See respective class documentation.
    ga_info, GAInfo class
        See respective class documentation.

    Returns
    -------
    pop, DEAP object
    stats, DEAP object
    hof, DEAP object

    See post_run function for examples of how to interpret results.

    """
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("indices", indiv_builder, gene_info)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", single_eval, gene_info)
    toolbox.register("mate", cxSDB, gene_info)
    toolbox.register("mutate", mutFlipper, gene_info)
    toolbox.register("select", tools.selTournament, tournsize=ga_info.nk)

    pop = toolbox.population(n=ga_info.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaSimple(pop, toolbox, ga_info.cxpb, ga_info.mutpb, ga_info.gen, stats, halloffame=hof)
    
    return pop, stats, hof

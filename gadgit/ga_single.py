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
    
def cx_SDB(gene_info, ind1, ind2):
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

def valid_add(gene_info, individual):
    """Based on gene info and current individual, return a valid index to add to an individual
    """
    return random.choice(list(set(range(0, gene_info.gene_count)) - individual))

def valid_remove(gene_info, individual):
    """Based on gene info, removed an index from an individual that respects fixed genes
    """
    return random.choice(sorted(tuple(individual - set(gene_info.fixed_list_ids))))

def self_correction(gene_info, individual):
    """This function takes a potentially broken individual and returns a correct one.

    Procedure:
        Add all fixed genes
        while size isn't right; add or remove
    """

    individual.update(gene_info.fixed_list_ids)
    while True:
        indiv_size = len(individual)
        if indiv_size < gene_info.com_size:
            individual.add(valid_add(gene_info, individual))
        elif indiv_size > gene_info.com_size:
            individual.remove(valid_remove(gene_info, individual))
        else: # Must be equal
            break

    assert len(individual) == gene_info.com_size, 'Self correction failed to create indiv with proper size'
    assert set(gene_info.fixed_list_ids).issubset(individual), 'Individual not possess all fixed genes after self correction'

    return individual

def cx_OPS(gene_info, ind1, ind2):
    """Standard one-point crossover implemented for set individuals.

    Self correction is handled by abstracted function.

    Note that this function has no ability to make assertions on the individuals it generates.
    """
    pivot = random.randint(0,gene_info.gene_count)
    ind1_new = [i for i in ind1 if i < pivot] # Read from same
    ind1_new.extend([i for i in ind2 if i > pivot]) # Read from other
    ind2_new = [i for i in ind2 if i < pivot]
    ind2_new.extend([i for i in ind1 if i > pivot])

    ind1.clear() # Forcibly use proper individual class
    ind1.update(ind1_new)
    ind2.clear()
    ind2.update(ind2_new)

    return self_correction(gene_info, ind1), self_correction(gene_info, ind2)

def mut_flipper(gene_info, individual):
    """Flip based mutation. Flip one off to on, and one on to off.
    
    Must not allow the choice of a fixed gene to be turned off.
    """
    assert len(individual) == gene_info.com_size, 'Mutation received invalid indiv'
    individual.remove(valid_remove(gene_info, individual))
    individual.add(valid_add(gene_info, individual))
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
    random.seed(ga_info.seed)

    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("indices", indiv_builder, gene_info)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", single_eval, gene_info)
    if ga_info.cross_meth == 'ops':
        toolbox.register("mate", cx_OPS, gene_info)
    elif ga_info.cross_meth == 'sdb':
        toolbox.register("mate", cx_SDB, gene_info)
    else:
        raise AttributeError('Invalid crossover string specified')
    toolbox.register("mutate", mut_flipper, gene_info)
    toolbox.register("select", tools.selTournament, tournsize=ga_info.nk)

    pop = toolbox.population(n=ga_info.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaSimple(pop, toolbox, ga_info.cxpb, ga_info.mutpb, ga_info.gen, stats, halloffame=hof)
    
    return pop, stats, hof

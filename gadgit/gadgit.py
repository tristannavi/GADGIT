import random
from collections.abc import Callable
from copy import deepcopy

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd
from numpy import ndarray
from scipy.stats import rankdata

from gadgit import GeneInfo, GAInfo

type pop = list[list]
type indiv = list


def cx_SDB(gene_info: GeneInfo, ind1: indiv, ind2: indiv) -> tuple[indiv, indiv]:
    """SDB Crossover

    Computes the intersection and asserts that after the intersection,
    the amount of genes left over to 'deal' between two new individuals
    is even.

    Clears the set structures of their old information, updates with the
    intersection, and lastly hands out half of the shuffled dealer to each
    indiv.

    ind1 and ind2 and kept as objects since they inherit from set, but have
    additional properties.

    Note that this process is not destructive to the fixed genes inside of the
    individuals.
    """

    # Build dealer
    s1 = set(ind1)
    s2 = set(ind2)
    intersect = s1.intersection(s2)
    dealer = list(s1.union(s2) - intersect)
    random.shuffle(dealer)
    assert len(dealer) % 2 == 0, 'Dealer assumption on indiv crossover failure'

    # Rebuild individuals and play out dealer
    ind1.clear()
    ind2.clear()
    ind1.extend(dealer[:len(dealer) // 2])
    ind1.extend(intersect)
    ind2.extend(dealer[len(dealer) // 2:])
    ind2.extend(intersect)
    assert (len(ind1) == gene_info.com_size and
            len(ind2) == gene_info.com_size), 'SDB created invalid individual'
    assert set(gene_info.fixed_list_ids).issubset(ind1), \
        'Ind1 does not possess all fixed genes after crossover'
    assert set(gene_info.fixed_list_ids).issubset(ind2), \
        'Ind2 does not possess all fixed genes after crossover'

    return ind1, ind2


def valid_add(gene_info: GeneInfo, individual: list) -> int:
    """
    Determines a valid addition to an individual's gene sequence from available genes.

    This function calculates a random valid gene not currently present in the given
    individual's gene sequence. The full range of gene indices is obtained from
    the `gene_info` object. It excludes the indices already existing in the `individual`
    sequence and randomly selects one from the remaining valid indices.

    :param gene_info: An object that contains information about available genes,
        including their total count in the `gene_count` attribute.
    :param individual: A list containing the indices of genes that
        are already part of the individual's gene sequence.
    :return: A randomly selected valid gene index that can be added
        to the individual's sequence.
    """
    return random.choice(list(set(range(0, gene_info.gene_count)) - set(individual)))


def valid_remove(gene_info: GeneInfo, individual: list) -> int:
    """Based on gene info, removed an index from an individual that respects
    fixed genes
    """
    return random.choice(sorted(tuple(set(individual) - set(gene_info.fixed_list_ids))))


def self_correction(gene_info: GeneInfo, individual: list) -> list:
    """This function takes a potentially broken individual and returns a
    correct one.

    Procedure:
        Add all fixed genes
        while size isn't right; add or remove
    """
    individual.extend(gene_info.fixed_list_ids)
    individual_new = list(set(individual))
    individual.clear()
    individual.extend(individual_new)
    while True:
        indiv_size = len(individual)
        if indiv_size < gene_info.com_size:
            individual.append(valid_add(gene_info, individual))
        elif indiv_size > gene_info.com_size:
            individual.remove(valid_remove(gene_info, individual))
        else:  # Must be equal
            break

    assert len(individual) == gene_info.com_size, \
        'Self correction failed to create indiv with proper size'
    assert set(gene_info.fixed_list_ids).issubset(individual), \
        'Individual not possess all fixed genes after self correction'

    return individual


def cx_OPS(gene_info: GeneInfo, ind1: list, ind2: list):
    """Standard one-point crossover implemented for set individuals.

    Self correction is handled by abstracted function.

    Note that this function has no ability to make assertions on the
    individuals it generates.
    """
    # pivot = random.randint(0, len(ind1) - 1)
    # ind1_new = [ind1[i] for i in range(0, pivot)]  # First part of the individual
    # ind1_new.extend([ind2[i] for i in range(pivot, len(ind2))])  # Second part of the other individual
    #
    # ind2_new = [ind2[i] for i in range(0, pivot)]
    # ind2_new.extend([ind1[i] for i in range(pivot, len(ind1))])
    #
    # ind1.clear()  # Forcibly use proper individual class
    # ind1.extend(ind1_new)
    # ind2.clear()
    # ind2.extend(ind2_new)

    cxpoint = random.randint(1, gene_info.com_size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return self_correction(gene_info, ind1), self_correction(gene_info, ind2)


def mut_flipper(gene_info, individual):
    """Flip based mutation. Flip one off to on, and one on to off.

    Must not allow the choice of a fixed gene to be turned off.
    """
    assert len(individual) == gene_info.com_size, \
        'Mutation received invalid indiv'
    individual.remove(valid_remove(gene_info, individual))
    individual.append(valid_add(gene_info, individual))
    assert len(individual) == gene_info.com_size, \
        'Mutation created an invalid indiv'
    assert set(gene_info.fixed_list_ids).issubset(individual), \
        ('Individual does not possess all fixed genes after mutation')

    return individual,


def indiv_builder(gene_info):
    """Implementation of forcing fixed genes in creation of new individual."""
    num_choices = gene_info.com_size - len(gene_info.fixed_list)
    valid_choices = list(set(range(gene_info.gene_count)) - set(gene_info.fixed_list_ids))
    base_indiv = random.sample(valid_choices, num_choices)
    base_indiv.extend(gene_info.fixed_list_ids)
    return base_indiv


def ga(gene_info: GeneInfo, ga_info: GAInfo, mapper: Callable = map, swap_meth: bool = False, **kwargs):
    """Main loop which sets DEAP objects and calls a multi objective EA algorithm.

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

    # random.seed(ga_info.seed)
    random.seed(ga_info.seed)

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("indices", indiv_builder, gene_info)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("map", mapper)

    if ga_info.cross_meth == 'ops':
        toolbox.register("mate", cx_OPS, gene_info)
    elif ga_info.cross_meth == 'sdb':
        toolbox.register("mate", cx_SDB, gene_info)
    elif ga_info.cross_meth == 'both':
        toolbox.register("mate_ops", cx_OPS, gene_info)
        toolbox.register("mate_sdb", cx_SDB, gene_info)
    else:
        raise AttributeError('Invalid crossover string specified')

    toolbox.register("mutate", mut_flipper, gene_info)
    toolbox.register("select", tools.selTournament, tournsize=ga_info.nk)

    pop = toolbox.population(n=ga_info.pop)
    # Empty, as SoR objects are special
    stats = tools.Statistics()
    extra_returns = None

    _, _, hof, extra_returns = ea_sum_of_ranks(ga_info, gene_info, pop, toolbox, ga_info.cxpb, ga_info.mutpb,
                                               ga_info.gen, stats, elite=[])

    return pop, stats, hof, extra_returns


def multi_eval(gene_info: GeneInfo, population: list[list[int]], *args) -> tuple[ndarray, dict]:
    """Helper function to implement the SoR table operations."""
    # Build raw objective information
    all_rows = np.zeros(shape=(len(population), len(gene_info.obj_list)))
    for index, indiv in enumerate(population):
        indiv_slice = gene_info.data_numpy[indiv]
        indiv_sums = indiv_slice.sum(axis=0)
        all_rows[index] = indiv_sums

    # Ranking procedure
    sor = np.zeros(shape=(len(population), len(gene_info.obj_list)))
    obj_log_info = {}
    for i, obj in enumerate(gene_info.obj_list):
        obj_log_info[f'new_gen_max_{obj}'] = all_rows[:, i].max()
        obj_log_info[f'new_gen_mean_{obj}'] = all_rows[:, i].mean()
        append_ranks = rankdata(all_rows[:, i], method="dense")
        # Normalize
        sor[:, i] = append_ranks / append_ranks.max()
    # Sum the ranks
    objective_sums = sor.sum(axis=1)
    return len(population) - rankdata(objective_sums, method="dense"), obj_log_info


def ea_sum_of_ranks(ga_info: GAInfo, gene_info: GeneInfo, population: list[base], toolbox, cxpb: float, mutpb: float,
                    ngen: int, stats=None, elite=None, verbose=__debug__, **kwargs):
    """
    This function runs an EA using the Sum of Ranks (SoR) fitness methodology.

    It is essentially a fork of the eaSimple function from deap.
    It is not meant to be exposed to users, and instead is only used
    internally by the package.
    """

    extra_returns: dict = {}

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']
    if stats:
        for obj in gene_info.obj_list:
            logbook.header.append(f'new_gen_max_{obj}')
            logbook.header.append(f'new_gen_mean_{obj}')

    # Offload SoR to table
    fit_series: np.ndarray
    fit_series, obj_log_info = multi_eval(gene_info, population, 0)

    # Update ALL fitness vals
    for index, fit_val in enumerate(fit_series):
        ## Single fitness value for the whole community (all genes in the community within one individual)
        population[index].fitness.values = fit_val,

    # elite = [deepcopy(population[fit_series.argmax()])]
    elite = [deepcopy(population[fit_series.argmin()])]
    # extra_returns.setdefault("elite_changed_temp", [])
    # extra_returns.setdefault("elite", [])
    # extra_returns["elite"].append(elite[0])
    #
    # extra_returns["elite_changed_temp"].append(elite[0].fitness.values[0])
    logbook.record(gen=0, nevals='maximal-temp', **obj_log_info)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals to breed
        # TODO: select pop-1 and add elite
        population.append(deepcopy(elite[0]))
        breed_pop = toolbox.select(population, len(population) - 1)

        offspring = varAnd(breed_pop, toolbox, cxpb, mutpb)
        offspring.append(deepcopy(elite[0]))

        # TODO maybe no mutation on elite or at least ensure elite is there for fitness calc

        # Offload SoR to table
        fit_series, obj_log_info = multi_eval(gene_info, offspring, gen)

        # Update ALL fitness vals
        for index, fit_val in enumerate(fit_series):
            offspring[index].fitness.values = fit_val,

        def string(list_of_ints):
            return ','.join(str(i) for i in list_of_ints)

        # Strict elitism

        # offspring.append(elite[0])

        # Update elite if a new individual either has a better fitness or the same fitness
        # Need to copy not reference!!
        elite = [
            # deepcopy(offspring[fit_series.argmax()]) if offspring[fit_series.argmax()].fitness.values[0] >= fit_series[offspring.index(elite[0])] else elite[0]]
            # deepcopy(offspring[fit_series.argmin()]) if offspring[fit_series.argmin()].fitness.values[0] <= fit_series[offspring.index(elite[0])] else elite[0]]
            deepcopy(offspring[fit_series.argmin()])]
        # extra_returns.setdefault("elite_changed_temp", [])
        extra_returns.setdefault("elite", [])
        # extra_returns["elite_changed_temp"].append(offspring[fit_series.argmax()].fitness.values[0])
        extra_returns["elite"].append(list(elite[0]))
        # elite.update(offspring)

        # if elite[0].fitness != offspring[fit_series.argmax()].fitness:
        #     extra_returns.setdefault("elite_fitnesses", 0)
        #     extra_returns["elite_fitnesses"] += 1
        # else:
        #     extra_returns.setdefault("elite_fitnesses_generation", [])
        #     extra_returns["elite_fitnesses_generation"].append(gen)

        # extra_returns.setdefault("elite", [None])
        # if extra_returns["elite"][-1] != elite[0]:
        #     extra_returns["elite"].append(elite[0])

        # Why select the best again and not only update? Selection occurs at the beginning of the loop
        population = offspring[:]

        # Update frontier based on elite index
        # How many times the gene has been seen in the elite community
        for index in elite[0]:
            gene_info.frontier[index] += 1

        # ranks[gen] = {
        #     "elite": elite[0],
        #     "frontier": gene_info.frontier,
        #     "fitness": fit_series,
        # }

        # # Manually marking old individuals
        # for indiv in population:
        #     del indiv.fitness.values

        # Append the current generation statistics to the logbook
        logbook.record(gen=gen, nevals='maximal-temp', **obj_log_info)
        if verbose:
            print(logbook.stream)

    # Check if elite ever happens to get smaller than a previous elite
    # extra_returns.setdefault("elite_changed", False)
    # for x in range(1, len(extra_returns["elite_changed_temp"])):
    #     if extra_returns["elite_changed_temp"][x] < extra_returns["elite_changed_temp"][x-1]:
    #         extra_returns["elite_changed"] = True
    #         break
    # del extra_returns["elite_changed_temp"]
    return population, logbook, elite, extra_returns

import random
from collections.abc import Callable
from typing import List

import numpy as np
import pandas as pd
from deap import base, algorithms
from deap import creator
from deap import tools
from deap.algorithms import varOr
from numpy import ndarray
from scipy.stats import rankdata

from gadgit import GeneInfo, GAInfo


def single_eval(gene_info, individual):
    """ Single objective summation of the centrality of a particular
    frame's chosen column.

    Due to gene_info.obj_list obviously accepting a list for the purposes of
    extending to MOP, in the case of this single_eval, the head of the list
    is treated as the 'single' objective.

    Note: does not correctly calculate the frontier.

    Largest sum is the best.
    """

    # assert len(individual) == gene_info.com_size, \
    #     'Indiv does not match community size in eval'
    # assert set(gene_info.fixed_list_ids).issubset(individual), \
    #     'Indiv does not possess all fixed genes'

    # FIXME Does this just sum the whole column?
    fit_col = gene_info.obj_list[0]
    fit_sum = 0.0
    for item in individual:
        fit_sum += gene_info.data_frame.loc[item, fit_col]
        # gene_info.frontier[item] += 1

    return fit_sum,


def cx_SDB(gene_info, ind1, ind2):
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
    intersect = ind1.intersection(ind2)
    dealer = list(ind1.union(ind2) - intersect)
    random.shuffle(dealer)
    assert len(dealer) % 2 == 0, 'Dealer assumption on indiv crossover failure'

    # Rebuild individuals and play out dealer
    ind1.clear()
    ind2.clear()
    ind1.update(dealer[:len(dealer) // 2])
    ind1.update(intersect)
    ind2.update(dealer[len(dealer) // 2:])
    ind2.update(intersect)
    assert (len(ind1) == gene_info.com_size and
            len(ind2) == gene_info.com_size), 'SDB created invalid individual'
    assert set(gene_info.fixed_list_ids).issubset(ind1), \
        'Ind1 does not possess all fixed genes after crossover'
    assert set(gene_info.fixed_list_ids).issubset(ind2), \
        'Ind2 does not possess all fixed genes after crossover'

    return ind1, ind2


def valid_add(gene_info, individual):
    """Based on gene info and current individual, return a valid index to add
    to an individual.
    """
    return random.choice(list(set(range(0, gene_info.gene_count)) - set(individual)))


def valid_remove(gene_info, individual: List):
    """Based on gene info, removed an index from an individual that respects
    fixed genes
    """
    return random.choice(sorted(tuple(set(individual) - set(gene_info.fixed_list_ids))))


def self_correction(gene_info, individual: List):
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
    # assert set(gene_info.fixed_list_ids).issubset(individual), \
    #     'Individual not possess all fixed genes after self correction'

    return individual


def cx_OPS(gene_info: GeneInfo, ind1: List, ind2: List):
    """Standard one-point crossover implemented for set individuals.

    Self correction is handled by abstracted function.

    Note that this function has no ability to make assertions on the
    individuals it generates.
    """
    pivot = random.randint(0, len(ind1) - 1)
    ind1_new = [ind1[i] for i in range(0, pivot)]  # First part of the individual
    ind1_new.extend([ind2[i] for i in range(pivot, len(ind2))])  # Second part of the other individual

    ind2_new = [ind2[i] for i in range(0, pivot)]
    ind2_new.extend([ind1[i] for i in range(pivot, len(ind1))])

    ind1.clear()  # Forcibly use proper individual class
    ind1.extend(ind1_new)
    ind2.clear()
    ind2.extend(ind2_new)

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
    valid_choices = list(set(range(gene_info.gene_count))
                         - set(gene_info.fixed_list_ids))
    base_indiv = random.sample(valid_choices, num_choices)
    base_indiv.extend(gene_info.fixed_list_ids)
    return base_indiv


def ga(gene_info, ga_info, mapper=map):
    if len(gene_info.obj_list) < 1:
        raise ValueError("You must have at least one objective to run a GA.")
    elif len(gene_info.obj_list) == 1:
        return ga_single(gene_info, ga_info)
    else:
        return ga_multi(gene_info, ga_info, mapper)


def ga_single(gene_info, ga_info):
    """Main loop which sets DEAP objects and calls a single objective EA algorithm.

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
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", single_eval, gene_info)

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
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    # FIXME does it matter if we use eaSimple or the multi-objective one?

    algorithms.eaSimple(pop, toolbox, ga_info.cxpb, ga_info.mutpb, ga_info.gen,
                        stats, halloffame=hof)

    # ea_sum_of_ranks(ga_info, gene_info, pop, toolbox, ga_info.cxpb, ga_info.mutpb,
    #                 ga_info.gen, stats, halloffame=hof)

    return pop, stats, hof


debug = False


def ga_multi(gene_info: GeneInfo, ga_info: GAInfo, mapper: Callable = map, swap_meth: bool = False, **kwargs):
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

    random.seed(ga_info.seed)

    creator.create("Fitness", base.Fitness, weights=(1.0,))
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

    _, _, hof = ea_sum_of_ranks(ga_info, gene_info, pop, toolbox, ga_info.cxpb, ga_info.mutpb,
                                ga_info.gen, stats, elite=None)

    return pop, stats, hof, extra_returns


extra_returns = {}


def multi(gene_info, population):
    """Helper function to implement the SoR table operations."""
    # Build raw objective information
    all_rows = []
    for indiv in population:
        indiv_slice = gene_info.data_frame.loc[indiv]
        indiv_sums = [indiv_slice[obj].sum() for obj in gene_info.obj_list]
        all_rows.append(indiv_sums)
    raw_frame = pd.DataFrame(all_rows, columns=gene_info.obj_list)

    # Ranking procedure
    sor = pd.DataFrame()
    obj_log_info = {}
    for obj in raw_frame.columns:
        obj_log_info[f'new_gen_max_{obj}'] = raw_frame[obj].max()
        obj_log_info[f'new_gen_mean_{obj}'] = raw_frame[obj].mean()
        rank_series = np.argsort(raw_frame[obj])
        # Flip index and argmax indices, create a series
        swap_index = pd.Series(dict((v, k) for k, v in rank_series.items()))
        # Sort by index (now argmax index)
        append_ranks = swap_index.sort_index()
        # Normalize
        sor[obj + '_rank_norm'] = append_ranks / append_ranks.max()
    # Sum the ranks
    sor['sum'] = sor[list(sor.columns)].sum(axis=1)
    # Rank the sums calculated previously
    # temp: pd.DataFrame
    # temp = sor['sum'].rank(method='first')

    rank_series2 = rank_series
    swap_index2 = swap_index
    append_ranks2 = append_ranks
    summation = sor[list(sor.columns)].sum(axis=1)
    first_ranking = sor['sum'].rank(method='first')
    append_ranks_max = append_ranks.max()
    append_ranks_min = append_ranks.min()
    single = [single_eval(gene_info, population[x]) for x in range(len(population))]
    single_min = np.array(single).flatten()[np.array(single).flatten().argmin()], np.array(
        single).flatten().argmin()
    single_max = np.array(single).flatten()[np.array(single).flatten().argmax()], np.array(
        single).flatten().argmax()

    return sor['sum'].rank(method='first'), sor["sum"]


def multi_eval(gene_info: GeneInfo, population: List[int], gen: int) -> tuple[ndarray, dict]:
    """Helper function to implement the SoR table operations."""
    # Build raw objective information
    all_rows = np.ndarray(shape=(len(population), len(gene_info.obj_list)))
    for index, indiv in enumerate(population):
        indiv_slice = gene_info.data_frame[gene_info.obj_list].to_numpy()[indiv]
        indiv_sums = indiv_slice.sum(axis=0)
        all_rows[index] = indiv_sums

    # Ranking procedure
    sor = np.ndarray(shape=(len(population), len(gene_info.obj_list)))
    obj_log_info = {}
    for i, obj in enumerate(gene_info.obj_list):
        obj_log_info[f'new_gen_max_{obj}'] = all_rows[:, i].max()
        obj_log_info[f'new_gen_mean_{obj}'] = all_rows[:, i].mean()
        rank_series = np.argsort(all_rows[:, i])
        # Flip index and argmax indices, create a series
        # Sort by index (now argmax index)
        append_ranks = np.arange(len(population))[np.argsort(all_rows[:, i]).argsort()].T
        # Normalize
        sor[:, i] = append_ranks / append_ranks.max()
    # Sum the ranks
    objective_sums = sor.sum(axis=1)
    # TODO: handle ties
    return rankdata(objective_sums), obj_log_info


def ea_sum_of_ranks(ga_info, gene_info: GeneInfo, population: list[base], toolbox, cxpb: float, mutpb: float,
                    ngen: int, stats=None, elite=None, verbose=__debug__, **kwargs):
    """
    This function runs an EA using the Sum of Ranks (SoR) fitness methodology.

    It is essentially a fork of the eaSimple function from deap.
    It is not meant to be exposed to users, and instead is only used
    internally by the package.
    """

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

    elite = [population[fit_series.argmax()]]

    logbook.record(gen=0, nevals='maximal-temp', **obj_log_info)
    if verbose:
        print(logbook.stream)

    def varOr2(population: list[base], toolbox, lambda_: int, cxpb: float, mutpb: float, gen: int, ngen: int,
               swap_percent: float) -> list[base]:
        """
            Implementation of a custom evolutionary algorithm that uses different crossover
            and mutation strategies based on the generation progress. It ensures the
            constraints on the sum of crossover and mutation probabilities and performs
            variation operations including crossover, mutation, and reproduction on a population.

            Taken from the DEAP library's algorithms.varOr function.

            Parameters
            ----------
            population : list
                A list of individuals representing the current generation's population.
            toolbox : object
                A DEAP toolbox containing crossover, mutation, and selection operators, among
                other evolutionary operators and utilities.
            lambda_ : int
                The number of offspring to generate, equivalent to the size of the next
                generation.
            cxpb : float
                The probability of applying the crossover operator during the variation
                process.
            mutpb : float
                The probability of applying the mutation operator during the variation
                process.
            gen : int
                The current generation number.
            ngen : int
                Number of total generations the evolutionary process is expected to run.

            Returns
            -------
            list
                A list of offspring generated after applying variation operations on the
                provided population.

            Raises
            ------
            AssertionError
                Raised if the sum of cxpb and mutpb exceeds 1.0.
        """
        assert (cxpb + mutpb) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")

        offspring = []
        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < cxpb:  # Apply crossover
                ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
                # Logic to switch crossover method based on generation progress
                if gen < ngen * swap_percent:
                    ind1, ind2 = toolbox.mate_ops(ind1=ind1, ind2=ind2)
                else:
                    ind1, ind2 = toolbox.mate_sdb(ind1=ind1, ind2=ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < cxpb + mutpb:  # Apply mutation
                ind = toolbox.clone(random.choice(population))
                ind, = toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
            else:  # Apply reproduction
                offspring.append(random.choice(population))

        return offspring

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals to breed
        # TODO: select pop-1 and add elite
        breed_pop = toolbox.select(population, len(population))

        offspring = varOr(breed_pop, toolbox, len(population), cxpb, mutpb)

        # TODO maybe no mutation on elite or at least ensure elite is there for fitness calc

        # Offload SoR to table
        fit_series, obj_log_info = multi_eval(gene_info, offspring, gen)

        # Update ALL fitness vals
        for index, fit_val in enumerate(fit_series):
            offspring[index].fitness.values = fit_val,

        # Strict elitism
        elite = [offspring[fit_series.argmax()]]
        population = tools.selBest(offspring + [elite[0]], len(population))

        # fit_series: pd.Series
        # fit_series.
        # Update frontier based on elite index
        ## How many times the gene has been seen?
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

    return population, logbook, elite

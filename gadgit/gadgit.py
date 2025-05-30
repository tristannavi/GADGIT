from collections.abc import Callable
from copy import deepcopy
import numba

import numpy as np
from numpy import copy
from numpy.typing import NDArray

from gadgit import GeneInfo, GAInfo


def cx_SDB(gene_info: GeneInfo, ind1: NDArray, ind2: NDArray):
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
    intersect = np.intersect1d(ind1, ind2, assume_unique=True)
    dealer = np.setdiff1d(np.union1d(ind1, ind2), intersect, assume_unique=True)
    gene_info.rand.shuffle(dealer)

    ind1[:len(dealer) // 2] = dealer[:len(dealer) // 2]
    ind1[len(dealer) // 2:] = intersect
    ind2[:len(dealer) // 2] = dealer[len(dealer) // 2:]
    ind2[len(dealer) // 2:] = intersect

    return ind1, ind2


def valid_add(gene_info: GeneInfo, individual: NDArray, len: int | None = None) -> int:
    """
    Determines a valid addition to an individual's gene sequence from available genes.

    This function calculates a random valid gene not currently present in the given
    individual's gene sequence. The full range of gene indices is obtained from
    the `gene_info` object. It excludes the indices already existing in the `individual`
    sequence and randomly selects one from the remaining valid indices.

    :param len:
    :param gene_info: An object that contains information about available genes,
        including their total count in the `gene_count` attribute.
    :param individual: A list containing the indices of genes that
        are already part of the individual's gene sequence.
    :return: A randomly selected valid gene index that can be added
        to the individual's sequence.
    """
    return gene_info.rand.choice(np.setdiff1d(np.arange(gene_info.gene_count), individual, assume_unique=True), len,
                                 replace=False)


def valid_remove(gene_info: GeneInfo, individual: NDArray, len: int | None = None) -> int:
    """Based on gene info, removed an index from an individual that respects
    fixed genes
    """
    return gene_info.rand.choice(np.nonzero(np.invert(np.isin(individual, gene_info.fixed_list_ids)))[0], len,
                                 replace=False)


def self_correction(gene_info: GeneInfo, individual: NDArray) -> NDArray:
    """This function takes a potentially broken individual and returns a
    correct one.

    Procedure:
        If the number of unique genes is lower than the required number,
        replace all duplicated genes with new ones
    """
    unique, unique_indices = np.unique(individual, return_index=True)
    mask = np.ones(len(individual), np.bool)
    mask[unique_indices] = 0
    # Too few genes
    if len(unique) < gene_info.com_size:
        individual[mask] = valid_add(gene_info, unique, gene_info.com_size - len(
            unique))

    return individual


def cx_OPS(gene_info: GeneInfo, ind1: NDArray, ind2: NDArray) -> tuple[NDArray, NDArray]:
    """
    Performs a crossover operation (CX-Operator Swap) between two individuals. This function modifies the genetic sequences
    of two parent individuals by swapping a segment of their genetic information at a randomly generated crossover point.
    The crossover point is determined by using the random integer generator from the provided gene information. After the
    swap, the individuals are adjusted by applying the self-correction function to ensure consistency or validity based
    on specific rules.

    :param gene_info: An instance of GeneInfo that contains genetic information and the random number generator used
        for determining the crossover point.
    :param ind1: The first individual's genetic sequence to participate in the crossover operation.
    :param ind2: The second individual's genetic sequence to participate in the crossover operation.
    :return: A tuple containing the modified genetic sequences of the two individuals after applying the crossover
        operation and subsequent self-correction.
    """

    cxpoint = gene_info.rand.integers(1, gene_info.com_size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = copy(ind2[cxpoint:]), copy(ind1[cxpoint:])

    return self_correction(gene_info, ind1), self_correction(gene_info, ind2)


def mut_flipper(gene_info: GeneInfo, individual: NDArray) -> NDArray:
    """
    Flips mutation values in a given genetic individual by identifying valid genes to
    remove and then determining valid genes to add. Updates the genetic individual
    accordingly.

    :param gene_info: Represents the gene information and constraints that
        determine which mutations are valid.
    :param individual: Numpy array representing a genetic individual's current state.
        This is modified in place based on valid mutation operations.
    :return: Updated genetic individual after performing mutations.
    """
    remove = valid_remove(gene_info, individual)
    individual[remove] = valid_add(gene_info, individual)

    return individual


def indiv_builder(gene_info: GeneInfo, pop_size: int) -> NDArray:
    """
    Generate a population of individuals based on gene information and population size.

    This function initializes a population of individuals represented as a numpy array
    of integers. Each individual is generated by combining a fixed list of gene IDs with
    a random selection of other gene IDs, based on provided gene information.

    :param gene_info: An object containing gene attributes, including the size of the
        chromosome, the fixed gene list, the count of possible genes, and the method to
        generate random numbers. This information is used to define the structure and
        constraints of each individual in the population.
    :param pop_size: The size of the population to generate. Determines how many
        individuals will be created.
    :return: A 2D numpy array representing the population, where each row is an
        individual and each column is a gene ID.
    """
    population = np.zeros(shape=(pop_size, gene_info.com_size), dtype=np.int64)
    num_choices = gene_info.com_size  # - len(gene_info.fixed_list)
    valid_choices = list(set(range(gene_info.gene_count)) - set(gene_info.fixed_list_ids))

    for i in range(pop_size):
        # base_indiv = np.zeros(shape=gene_info.com_size)#np.pad(gene_info.fixed_list_ids, (0, num_choices), 'constant')
        base_indiv = gene_info.rand.choice(valid_choices, num_choices, replace=False)
        # gene_info.rand.shuffle(base_indiv)
        population[i] = base_indiv

    return population


def tournament_selection(gene_info: GeneInfo, individuals: NDArray, k: int, tournsize: int,
                         fitnesses: NDArray, max: bool = False) -> NDArray:
    """
    Select individuals from a population using a tournament selection process.

    This function uses a stochastic method to select individuals from a given
    population (individuals) based on their fitness values (fitnesses). A subset
    of individuals (aspirants) is selected at random, and the fittest or least
    fit individual from this subset is chosen, depending on the maximization or
    minimization goal. The process continues until the desired number of
    individuals (k) is selected. Forked from DEAP.

    :param gene_info: Object containing a random number generator for selecting
        aspirants.
    :param individuals: Array representing the population of individuals to
        select from.
    :param k: Number of individuals to select from the population.
    :param tournsize: Number of aspirants in each tournament.
    :param fitnesses: Array of fitness values corresponding to the individuals.
    :param max: Indicator for selection goal; True for maximizing fitness and
        False for minimizing fitness. Defaults to False.
    :return: Array of selected individuals from the population based on the
        tournament selection method.
    """
    chosen = np.zeros_like(individuals)
    for i in range(k):
        aspirants = gene_info.rand.choice(np.arange(0, len(individuals)), tournsize, replace=False)
        if max:
            chosen[i] = individuals[aspirants][fitnesses[aspirants].argmax()]
        else:
            chosen[i] = individuals[aspirants][fitnesses[aspirants].argmin()]
    return chosen


def ga(gene_info: GeneInfo, ga_info: GAInfo, **kwargs):
    """
    This function executes a genetic algorithm (GA) for optimization, using a specified crossover
    method defined in the `GAInfo` argument. The genetic algorithm starts by creating an initial
    population and then evolves it over several generations according to the given parameters.
    The function determines the crossover method to be used based on the string provided in
    `ga_info.cross_meth`. The genetic algorithm uses the `ea_sum_of_ranks` function to evaluate
    the population over generations.

    :param gene_info: Information about the genes used in the genetic algorithm.
    :param ga_info: Contains configuration data for the genetic algorithm, including
                    crossover method, population size, mutation/crossover probabilities,
                    and the number of generations.
    :param kwargs: Optional keyword arguments that can be passed to the genetic
                   algorithm or related utility functions.
    :return: A tuple containing:
             - **pop**: Final evolved population from the last generation.
             - **dict**: Placeholder for additional data (currently empty).
             - **hof**: The best entities (Hall of Fame) identified by the genetic algorithm.
             - **extra_returns**: Any additional data returned by the execution.
    """
    if ga_info.cross_meth == 'ops':
        cross_meth = cx_OPS
    elif ga_info.cross_meth == 'sdb':
        cross_meth = cx_SDB
    # elif ga_info.cross_meth == 'both':
    else:
        raise AttributeError('Invalid crossover string specified')

    pop = indiv_builder(gene_info, ga_info.pop)

    pop, _, hof, extra_returns = ea_sum_of_ranks(ga_info, gene_info, pop, ga_info.cxpb, ga_info.mutpb, ga_info.gen,
                                                 cross_meth, kwargs=kwargs)

    return pop, {}, hof, extra_returns


@numba.njit
def _dense_rank(a: NDArray) -> NDArray:
    # returns 1-based dense ranks of 1D array a
    unique = np.unique(a)
    ranks = np.empty(a.shape, np.int64)
    for i in range(a.size):
        # linear search over unique (ok for moderate sizes)
        ai = a[i]
        for j in range(unique.size):
            if ai == unique[j]:
                ranks[i] = j + 1
                break
    return ranks


@numba.njit
def multi_eval_nb(data: NDArray,
                  population: NDArray,
                  fixed: NDArray,
                  minimize: np.bool = False
                  ) -> NDArray:
    pop_size, genome_len = population.shape
    num_objs = data.shape[1]

    all_rows = np.zeros((pop_size, num_objs))
    # build raw sums
    for i in range(pop_size):
        for g in range(genome_len):
            idx = population[i, g]
            for o in range(num_objs):
                all_rows[i, o] += data[idx, o]
                all_rows[i, o] += fixed[o]

    # prepare output array
    sor = np.zeros_like(all_rows)
    for o in range(num_objs):
        ranks = _dense_rank(all_rows[:, o])
        max_r = ranks.max()
        for i in range(pop_size):
            sor[i, o] = ranks[i] / max_r

    # sum over objectives and final rank
    obj_sums = np.sum(sor, axis=1)
    final_ranks = _dense_rank(obj_sums)

    if minimize:
        return len(population) - final_ranks
    else:
        return final_ranks


def varAnd(offspring: NDArray, cxpb: float, mutpb: float, gene_info: GeneInfo, cross_meth_func: Callable,
           pop_size: int):
    """
    Apply crossover and mutation on a given population of offspring based on the provided parameters.
    The function iterates over the population, performing crossover on adjacent individuals with a
    probability determined by `cxpb`. It also applies mutation to individuals with a probability
    determined by `mutpb`. These genetic operations help to introduce diversity and guide
    the population towards better solutions in evolutionary algorithms. Forked from DEAP.

    :param offspring: The population of individuals to which crossover and mutation will be applied.
    :param cxpb: The probability of performing crossover between two individuals.
    :param mutpb: The probability of mutating an individual in the population.
    :param gene_info: Instance containing metadata and tools needed for genetic operations,
        such as randomness generator and gene structure.
    :param cross_meth_func: Function responsible for executing crossover between two individuals.
    :param pop_size: Size of the population, indicating the number of individuals in `offspring`.

    :return: The modified population of individuals after applying crossover and mutation operations.
    """
    for i in range(1, pop_size, 2):
        if gene_info.rand.random() < cxpb:
            c1, c2 = cross_meth_func(gene_info, offspring[i - 1], offspring[i])
            offspring[i] = c1
            offspring[i - 1] = c2

    for i in range(pop_size):
        if gene_info.rand.random() < mutpb:
            offspring[i] = mut_flipper(gene_info, offspring[i])

    return offspring


def ea_sum_of_ranks(ga_info: GAInfo, gene_info: GeneInfo, population: NDArray, cxpb: float, mutpb: float, ngen: int,
                    cross_meth: Callable, **kwargs):
    """
    This function runs an EA using the Sum of Ranks (SoR) fitness methodology.

    It is essentially a fork of the eaSimple function from deap.
    It is not meant to be exposed to users, and instead is only used
    internally by the package.
    """
    extra_returns: dict = {}

    # Offload SoR to table
    fit_series: NDArray
    fit_series = multi_eval_nb(gene_info.data_numpy, population, gene_info.sum)

    elite = [deepcopy(population[fit_series.argmax()])]
    # elite = [deepcopy(population[fit_series.argmin()])]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if gen % 10 == 0:
            print(gen)
        # Select the next generation individuals to breed
        # TODO: select pop-1 and add elite
        breed_pop = tournament_selection(gene_info, population, len(population) - 1, ga_info.nk, fit_series, max=True)

        offspring = varAnd(breed_pop, cxpb, mutpb, gene_info, cross_meth, len(population) - 1)

        # Strict elitism
        offspring[len(population) - 1] = deepcopy(elite[0])

        # TODO maybe no mutation on elite or at least ensure elite is there for fitness calc

        # Offload SoR to table
        fit_series = multi_eval_nb(gene_info.data_numpy, offspring, gene_info.sum)

        # Update elite if a new individual either has a better fitness or the same fitness
        # Need to copy not reference!!
        # best_offspring_fitness = offspring[fit_series.argmin()].fitness.values[0]
        # elite_fitness = fit_series[offspring.index(elite[0])]

        best_current = fit_series.argmax()
        current_elite_fitness = fit_series[np.where((offspring == elite[0]).all(1))[0][0]]
        elite = [deepcopy(offspring[fit_series.argmax()]) if best_current >= current_elite_fitness else elite[0]]
        # deepcopy(offspring[fit_series.argmin()])]
        extra_returns.setdefault("elite", [])
        extra_returns["elite"].append(list(elite[0]))

        population = offspring

        # Update frontier based on elite index
        # How many times the gene has been seen in the elite community
        gene_info.frontier[elite[0]] += 1

    return population, {}, elite, extra_returns

from collections.abc import Callable
from copy import deepcopy
import numpy as np
from numpy import copy
from numpy.typing import NDArray
from gadgit.GeneInfo import GeneInfo
from gadgit.GAInfo import GAInfo


def cx_SDB(gene_info: GeneInfo, ind1: NDArray, ind2: NDArray):
    """
    Performs the Safe Dealer Based crossover
    algorithm operator between two parent individuals represented by their
    genetic code. The CX-SDB operator exchanges genetic information between the
    two parents by identifying common genes, separating unique genes, shuffling
    the unique genes, and recombining them into the offspring. It ensures a mix
    of genetic material while keeping consistency with the genetic structure.

    :param gene_info: An object that provides methods and properties for handling
        genetic operations, including a random number generator for shuffling
        operations.
    :param ind1: An array representing the first individual. It serves as the
        first parent and is modified in place to reflect one of the offspring of
        the crossover.
    :param ind2: An array representing the second individual. It serves as the
        second parent and is modified in place to reflect one of the offspring of
        the crossover.
    :return: A tuple containing two arrays. The first array corresponds to the
        modified version of `ind1` after the crossover, and the second array
        corresponds to the modified version of `ind2`.
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


def valid_add(gene_info: GeneInfo, individual: NDArray, count: int | None = None) -> int:
    """
    Determines a valid addition to an individual's gene sequence from available genes.

    This function calculates a random valid gene not currently present in the given
    individual's gene sequence. The full range of gene indices is obtained from
    the `gene_info` object. It excludes the indices already existing in the `individual`
    sequence and randomly selects one from the remaining valid indices.

    :param count: The number of valid genes to select. Defaults to 1.
    :param gene_info: An object that contains information about available genes,
        including their total count in the `gene_count` attribute.
    :param individual: A list containing the indices of genes that
        are already part of the individual's gene sequence.
    :return: A randomly selected valid gene index that can be added
        to the individual's sequence.
    """
    return gene_info.rand.choice(
        np.setdiff1d(np.setdiff1d(np.arange(gene_info.gene_count), individual, assume_unique=True),
                     gene_info.fixed_list_nums), count,
        replace=False)


def valid_remove(gene_info: GeneInfo, individual: NDArray, count: int | None = None) -> int:
    return gene_info.rand.choice(len(individual), count, replace=False)


def self_correction(gene_info: GeneInfo, individual: NDArray) -> NDArray:
    """
    Performs a self-correction operation on an individual's gene representation to
    ensure that it meets the required number of unique components. If the unique
    genes within the individual are fewer than the required `com_size` defined in
    `gene_info`, the missing genes are replaced accordingly.

    This function identifies the unique genes within the individual and their indices in the original array.
    A mask is used to ensure that the genes do not get sorted, and only ones that should be changed will be changed.

    :param gene_info: Contains information about the gene, including size and
        configuration constraints required for the individual.
    :param individual: An array-like structure representing the individual's genes.
        The array contains gene indices that define this particular individual.
    :return: A corrected version of the individual's genes, ensuring that it
        contains the required number of unique components as specified in
        `gene_info`.
    """

    unique, unique_indices = np.unique(individual, return_index=True)
    mask = np.ones(len(individual), np.bool)
    mask[unique_indices] = 0
    # If there are too few genes in the unique array, add more
    if len(unique) < gene_info.com_size:
        individual[mask] = valid_add(gene_info, unique, gene_info.com_size - len(unique))

    return individual


def cx_OPS(gene_info: GeneInfo, ind1: NDArray, ind2: NDArray) -> tuple[NDArray, NDArray]:
    """
    Performs a crossover operation (CX-Operator Swap) between two individuals. This function modifies the genetic sequences
    of two parent individuals by swapping a segment of their genetic information at a randomly generated crossover point.
    The crossover point is determined by using the random integer generator from the provided gene information. After the
    swap, the individuals are adjusted by applying the self-correction function to ensure consistency or validity based
    on specific rules. Forked from DEAP.

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


def population_builder(gene_info: GeneInfo, pop_size: int) -> NDArray:
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
    valid_choices = list(set(range(gene_info.gene_count)) - set(gene_info.fixed_list_nums))

    for i in range(pop_size):
        individual = gene_info.rand.choice(valid_choices, gene_info.com_size, replace=False)
        population[i] = individual

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


def _rank(array: NDArray, minimize: bool = True) -> NDArray:
    """
    Calculate 1-based dense ranks of elements in a 1D array.

    The function computes the ranks of elements in the input array such that
    equal elements receive the same rank, and ranks are assigned consecutively
    from 1 to the number of unique elements.

    The implementation uses a linear search over the unique elements of
    the input array for each element, making it suitable for moderate-sized
    arrays.

    :param array: The input 1D array for which ranks are to be computed.
                  It should be a numpy array.
    :return: A 1D numpy array containing the 1-based dense ranks of the
             elements in the input array. The length of the result array
             matches the length of the input array.
    """
    # `_`  -> sorted unique values
    # `inv`-> for every element of `a` the index of the matching unique value
    _, inv = np.unique(array, return_inverse=True)
    if not minimize:
        inv += 1
        return np.max(inv) + 1 - inv
    return inv + 1  # make the ranks 1-based instead of 0-based


def multi_eval_nb(data: NDArray,
                  population: NDArray,
                  fixed: NDArray,
                  minimize: bool = False) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Evaluates and ranks a population based on given data, a fixed vector, and whether the ranking is
    maximization-oriented. The evaluation involves building raw sums for selected genes, calculating
    normalized ranks for each objective, and combining the objectives' scores into final ranks.

    :param data: A Numpy array where each row corresponds to an individual centrality vector, and
        each column represents an objective.
    :param population: A Numpy 2D array where each row represents an individual in the population
        and columns correspond to the genes (selected indices into `data` for evaluation).
    :param fixed: A Numpy 1D array containing fixed values to add to the objective scores of
        each individual. This array must have a length equal to the number of objectives.
    :param minimize: A boolean indicating whether the ranking should interpret higher scores as
        better. If True, the ranking is reversed such that higher scores are better; if False,
        lower scores are better.
    :return: A Numpy 1D array where each value corresponds to the rank of the respective individual
        in the input `population`. Ranks are calculated based on combined normalized scores of
        all objectives. Ranks are in ascending order if `maximize` is False, and descending
        order otherwise.
    """
    pop_size, com_size = population.shape
    num_objs = data.shape[1]
    all_rows = np.zeros(shape=(pop_size, num_objs))
    max_values = np.zeros(num_objs)
    avg_values = np.zeros(num_objs)
    min_values = np.zeros(num_objs)

    # Sum the centralities for every gene in each individual for each objective
    for gene in range(com_size):
        centrality_indices = population[:, gene]
        all_rows += data[centrality_indices]

    # Add the centrality measures of the fixed genes
    for objective in range(num_objs):
        all_rows[:, objective] += fixed[objective]
        max_values[objective] = all_rows[:, objective].max()
        avg_values[objective] = all_rows[:, objective].mean()
        min_values[objective] = all_rows[:, objective].min()

    # Rank each individual based on the sum of their centralities
    sor = np.zeros_like(all_rows)
    for objective in range(num_objs):
        ranks = _rank(all_rows[:, objective], minimize)
        max_rank = ranks.max()
        for individual_index in range(pop_size):
            sor[individual_index, objective] = ranks[individual_index] / max_rank

    # Sum all objective ranks for each individual and then rank the individuals based on those sums
    obj_sums = np.sum(sor, axis=1)
    final_ranks = _rank(obj_sums)

    return final_ranks, max_values, avg_values, min_values
    # return all_rows, max_values, avg_values, min_values


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
            offspring[i - 1], offspring[i] = cross_meth_func(gene_info, offspring[i - 1], offspring[i])

    for i in range(pop_size):
        if gene_info.rand.random() < mutpb:
            offspring[i] = mut_flipper(gene_info, offspring[i])

    return offspring


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

    pop = population_builder(gene_info, ga_info.pop)
    logs = None

    pop, log, hof, extra_returns = ea_sum_of_ranks(ga_info, gene_info, pop, ga_info.cxpb, ga_info.mutpb, ga_info.gen,
                                                   cross_meth, kwargs=kwargs)

    return pop, log, hof, extra_returns


def ea_sum_of_ranks(ga_info: GAInfo, gene_info: GeneInfo, population: NDArray, cxpb: float, mutpb: float, ngen: int,
                    cross_meth: Callable, kwargs):
    """
    This function runs an EA using the Sum of Ranks (SoR) fitness methodology.

    It is essentially a fork of the eaSimple function from deap.
    It is not meant to be exposed to users, and instead is only used
    internally by the package.
    """
    extra_returns: dict = {}
    gen = 0

    # Offload SoR to table
    fit_series: NDArray
    fit_series, max_fitness, avg_fitness, min_fitness = multi_eval_nb(gene_info.data_numpy, population, gene_info.sum)
    gene_counts = 0  # np.sum(population == kwargs.setdefault("loo_gene", ""))
    print("Gen:", gen, "Avg Fitness:", avg_fitness, "Max Fitness:", max_fitness, "Min Fitness:", min_fitness, "Unique:",
          len(np.unique(population)), "Count:", gene_counts)
    # print(f"{0}, {max_fitness[0]}, {len(np.unique(population))}")

    log: NDArray = np.zeros(shape=(ngen + 1, 3 * len(gene_info.obj_list) + 1 + 2))
    log[gen] = [gen, *avg_fitness, *max_fitness, *min_fitness, len(np.unique(population)), gene_counts]

    elite = [deepcopy(population[fit_series.argmin()])]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals to breed
        # breed_pop = tournament_selection(gene_info, population, len(population) - 1, ga_info.nk, fit_series)
        breed_pop = tournament_selection(gene_info, population, len(population) - 1, ga_info.nk, fit_series)

        population = varAnd(breed_pop, cxpb, mutpb, gene_info, cross_meth, len(population) - 1)

        # Strict elitism
        population[len(population) - 1] = deepcopy(elite[0])

        # Offload SoR to table
        fit_series, max_fitness, avg_fitness, min_fitness = multi_eval_nb(gene_info.data_numpy, population,
                                                                          gene_info.sum)
        gene_counts = 0  # np.sum(population == kwargs.setdefault("loo_gene", ""))
        print("Gen:", gen, "Avg Fitness:", avg_fitness, "Max Fitness:", max_fitness, "Min Fitness:", min_fitness,
              "Unique:", len(np.unique(population)), "Count:", gene_counts)
        # print(f"{0}, {max_fitness[0]}, {len(np.unique(population))}")

        log[gen] = [gen, *avg_fitness, *max_fitness, *min_fitness, len(np.unique(population)), gene_counts]

        # Update elite if a new individual either has a better fitness or the same fitness
        # Need to copy not reference!!
        elite = [deepcopy(population[fit_series.argmin()])]
        # elite = [deepcopy(population[fit_series.argmax()])]

        # extra_returns.setdefault("elite", [])
        # elite_list = list(elite[0])
        # if elite_list not in extra_returns["elite"]:
        #     extra_returns["elite"].append(elite_list)

        # Update frontier based on elite index
        # How many times the gene has been seen in the elite community
        gene_info.frontier[elite[0]] += 1

    return population, log, elite, extra_returns

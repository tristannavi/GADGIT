from collections.abc import Callable
from copy import deepcopy
import numba

import numpy as np

from gadgit import GeneInfo, GAInfo


def cx_SDB(gene_info: GeneInfo, ind1: np.ndarray, ind2: np.ndarray):
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
    # assert (len(ind1) == gene_info.com_size and
    #         len(ind2) == gene_info.com_size), 'SDB created invalid individual'
    # assert set(gene_info.fixed_list_ids).issubset(ind1), \
    #     'Ind1 does not possess all fixed genes after crossover'
    # assert set(gene_info.fixed_list_ids).issubset(ind2), \
    #     'Ind2 does not possess all fixed genes after crossover'

    return ind1, ind2


def valid_add(gene_info: GeneInfo, individual: np.ndarray) -> int:
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
    return gene_info.rand.choice(np.setdiff1d(np.arange(gene_info.gene_count), individual, assume_unique=True))


def valid_remove(gene_info: GeneInfo, individual: np.ndarray) -> int:
    """Based on gene info, removed an index from an individual that respects
    fixed genes
    """
    return gene_info.rand.choice(np.nonzero(np.invert(np.isin(individual, gene_info.fixed_list_ids)))[0])


def self_correction(gene_info: GeneInfo, individual: np.ndarray) -> np.ndarray:
    """This function takes a potentially broken individual and returns a
    correct one.

    Procedure:
        Add all fixed genes
        while size isn't right; add or remove
    """
    individual = np.unique(np.append(individual, gene_info.fixed_list_ids))
    if gene_info.com_size - len(individual) > 0:
        temp = np.zeros(shape=(gene_info.com_size - len(individual)))
        for x in range(len(temp)):
            temp[x] = valid_add(gene_info, individual)
        return np.append(individual, temp)
    elif gene_info.com_size - len(individual) < 0:
        for _ in range(abs(gene_info.com_size - len(individual))):
            individual[valid_remove(gene_info, individual)] = -1
        return np.delete(individual, np.where(individual == -1))

    # assert len(individual) == gene_info.com_size, \
    #     'Self correction failed to create indiv with proper size'
    # assert set(gene_info.fixed_list_ids).issubset(individual), \
    #     'Individual not possess all fixed genes after self correction'

    return individual


def cx_OPS(gene_info: GeneInfo, ind1: np.ndarray, ind2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard one-point crossover implemented for set individuals.

    Self correction is handled by abstracted function.

    Note that this function has no ability to make assertions on the
    individuals it generates.
    """

    cxpoint = gene_info.rand.integers(1, gene_info.com_size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return self_correction(gene_info, ind1), self_correction(gene_info, ind2)


def mut_flipper(gene_info: GeneInfo, individual: np.ndarray) -> np.ndarray:
    """Flip based mutation. Flip one off to on, and one on to off.

    Must not allow the choice of a fixed gene to be turned off.
    """
    # assert len(individual) == gene_info.com_size, \
    #     'Mutation received invalid indiv'
    remove = valid_remove(gene_info, individual)
    individual[remove] = valid_add(gene_info, individual)
    # assert len(individual) == gene_info.com_size, \
    #     'Mutation created an invalid indiv'
    # assert set(gene_info.fixed_list_ids).issubset(individual), \
    #     ('Individual does not possess all fixed genes after mutation')

    return individual


def indiv_builder(gene_info: GeneInfo) -> np.ndarray:
    """Implementation of forcing fixed genes in creation of new individual."""
    num_choices = gene_info.com_size - len(gene_info.fixed_list)
    valid_choices = list(set(range(gene_info.gene_count)) - set(gene_info.fixed_list_ids))
    base_indiv = np.pad(gene_info.fixed_list_ids, (0, num_choices), 'constant')
    base_indiv[len(gene_info.fixed_list_ids):] = gene_info.rand.choice(valid_choices, num_choices, replace=False)
    return base_indiv


def tournament_selection(gene_info: GeneInfo, individuals: np.ndarray, k: int, tournsize: int,
                         fitneses: np.ndarray, max: bool = False) -> np.ndarray:
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = np.zeros_like(individuals)
    for i in range(k):
        aspirants = gene_info.rand.choice(np.arange(0, len(individuals)), tournsize, replace=False)
        if max:
            chosen[i] = individuals[aspirants][fitneses[aspirants].argmax()]
        else:
            chosen[i] = individuals[aspirants][fitneses[aspirants].argmin()]
    return chosen


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

    if ga_info.cross_meth == 'ops':
        cross_meth = cx_OPS
    elif ga_info.cross_meth == 'sdb':
        cross_meth = cx_SDB
    # elif ga_info.cross_meth == 'both':
    else:
        raise AttributeError('Invalid crossover string specified')

    pop = np.array([indiv_builder(gene_info) for _ in range(ga_info.pop)])

    _, _, hof, extra_returns = ea_sum_of_ranks(ga_info, gene_info, pop, ga_info.cxpb, ga_info.mutpb,
                                               ga_info.gen, cross_meth, elite=[])

    return pop, {}, hof, extra_returns


@numba.njit
def _dense_rank(a: np.ndarray) -> np.ndarray:
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
def multi_eval_nb(data: np.ndarray,
                  population: np.ndarray
                  ) -> np.ndarray:
    pop_size, genome_len = population.shape
    num_objs = data.shape[1]

    all_rows = np.zeros((pop_size, num_objs))
    # build raw sums
    for i in range(pop_size):
        for g in range(genome_len):
            idx = population[i, g]
            for o in range(num_objs):
                all_rows[i, o] += data[idx, o]

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

    return len(population) - final_ranks


def varAnd(offspring: np.ndarray, cxpb: float, mutpb: float, gene_info: GeneInfo, cross_meth_func: Callable):
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if gene_info.rand.random() < cxpb:
            c1, c2 = cross_meth_func(gene_info, offspring[i - 1], offspring[i])
            offspring[i] = c1
            offspring[i - 1] = c2

    for i in range(len(offspring)):
        if gene_info.rand.random() < mutpb:
            offspring[i] = mut_flipper(gene_info, offspring[i])

    return offspring


def ea_sum_of_ranks(ga_info: GAInfo, gene_info: GeneInfo, population: np.ndarray, cxpb: float,
                    mutpb: float, ngen: int, cross_meth: Callable, elite=None, **kwargs):
    """
    This function runs an EA using the Sum of Ranks (SoR) fitness methodology.

    It is essentially a fork of the eaSimple function from deap.
    It is not meant to be exposed to users, and instead is only used
    internally by the package.
    """
    extra_returns: dict = {}

    # Offload SoR to table
    fit_series: np.ndarray
    fit_series = multi_eval_nb(gene_info.data_numpy, population)

    # elite = [deepcopy(population[fit_series.argmax()])]
    elite = [deepcopy(population[fit_series.argmin()])]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if gen % 10 == 0:
            print(gen)
        # Select the next generation individuals to breed
        # TODO: select pop-1 and add elite
        breed_pop = tournament_selection(gene_info, population, len(population), ga_info.nk, fit_series)

        offspring = varAnd(breed_pop, cxpb, mutpb, gene_info, cross_meth)
        offspring[0] = deepcopy(elite[0])

        # TODO maybe no mutation on elite or at least ensure elite is there for fitness calc

        # Offload SoR to table
        fit_series = multi_eval_nb(gene_info.data_numpy, offspring)

        # Strict elitism

        # Update elite if a new individual either has a better fitness or the same fitness
        # Need to copy not reference!!
        # best_offspring_fitness = offspring[fit_series.argmin()].fitness.values[0]
        # elite_fitness = fit_series[offspring.index(elite[0])]
        elite = [
            # deepcopy(offspring[fit_series.argmax()]) if offspring[fit_series.argmax()].fitness.values[0] >= fit_series[offspring.index(elite[0])] else elite[0]]
            # deepcopy(offspring[fit_series.argmin()]) if offspring[fit_series.argmin()].fitness.values[0] <= fit_series[offspring.index(elite[0])] else elite[0]]
            deepcopy(offspring[fit_series.argmin()])]
        extra_returns.setdefault("elite", [])
        extra_returns["elite"].append(list(elite[0]))

        population = offspring

        # Update frontier based on elite index
        # How many times the gene has been seen in the elite community
        gene_info.frontier[elite[0]] += 1

    return population, {}, elite, extra_returns

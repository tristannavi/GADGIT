import numpy as np
from numpy.ma.testutils import assert_array_equal

import gadgit.gadgit
from gadgit import GeneInfo
from test_datasets import get_gene_info_two_attributes, get_gene_info_one_attribute, get_population, individuals


class Test:

    def test_fitness_duplicate_individual_two_attributes(self):
        gene_info: GeneInfo = get_gene_info_two_attributes()
        population = get_population(True)
        sum: np.float64 = gene_info.data_frame[gene_info.data_frame["GeneName"].isin(gene_info.fixed_list)][
            gene_info.obj_list].sum()
        population2 = np.zeros((len(population), gene_info.gene_count))
        for p in range(len(population)):
            population2[p][population[p]] = 1
        assert_array_equal(
            gadgit.gadgit.multi_eval_nb(np.array(gene_info.data_frame[gene_info.obj_list]), np.array(population2))[0],
            [7, 7, 4, 1, 2, 8, 9, 6, 3, 5])

    def test_fitness_duplicate_individual_one_attribute(self):
        gene_info: GeneInfo = get_gene_info_one_attribute()
        population = get_population(True)
        sum: np.float64 = gene_info.data_frame[gene_info.data_frame["GeneName"].isin(gene_info.fixed_list)][
            gene_info.obj_list].sum()
        population2 = np.zeros((len(population), gene_info.gene_count))
        for p in range(len(population)):
            population2[p][population[p]] = 1
        assert_array_equal(
            gadgit.gadgit.multi_eval_nb(np.array(gene_info.data_frame[gene_info.obj_list]), np.array(population2))[0],
            [9, 9, 6, 1, 3, 7, 8, 4, 2, 5])

    def test_indiv_builder(self):
        gene_info = get_gene_info_one_attribute()
        indiv = gadgit.gadgit.indiv_builder(gene_info, gene_info.com_size)
        assert 5 not in indiv
        assert 10 not in indiv
        assert len(indiv) == gene_info.com_size

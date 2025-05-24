import random

import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal

import gadgit.gadgit
from test_datasets import get_gene_info_two_attributes, get_gene_info_one_attribute, get_population

# random.seed(1)
class Test:
    @pytest.fixture
    def setup(self):
        self.two_attributes = get_gene_info_two_attributes()
        self.one_attribute = get_gene_info_one_attribute()

    @pytest.mark.usefixtures("setup")
    def test_multi_eval_duplicate_individual(self):
        assert_array_equal(gadgit.gadgit.multi_eval(self.two_attributes, get_population(True)),
                           [7, 7, 4, 1, 2, 8, 9, 6, 3, 5])

        assert_array_equal(gadgit.gadgit.multi_eval(self.one_attribute, get_population(True)),
                           [9, 9, 6, 1, 3, 7, 8, 4, 2, 5])

    def test_indiv_builder(self, setup):
        indiv = gadgit.gadgit.indiv_builder(self.one_attribute)
        assert 5 in indiv
        assert 10 in indiv
        assert len(indiv) == 10

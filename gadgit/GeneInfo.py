from typing import List

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, PCG64DXSM, Generator
from pandas import DataFrame


class GeneInfo:
    """
    A class to manage genetic information and parameters for evolutionary algorithms.

    This class provides tools to handle genetic datasets, configure genetic algorithm (GA)
    parameters, and compute relevant values for problem-solving using genetic algorithms.
    It initializes with a dataset and specified objectives, allowing for fixed genes
    and specifying candidate community sizes.

    :ivar frame_path: The path or DataFrame containing genetic data.
    :ivar data_frame: A pandas DataFrame loaded from the frame_path.
    :ivar data_numpy: Numpy representation of the selected objectives.
    :ivar gene_count: Total number of genes in the dataset.
    :ivar obj_list: List of objectives to be used in the genetic algorithm.
    :ivar fixed_list: List of genes to keep fixed in candidate solutions.
    :ivar fixed_list_nums: Numpy array of indices of fixed genes within the dataset.
    :ivar com_size: Size of the candidate community, adjusted by the fixed list.
    :ivar indiv_len: Numpy array representing indices within the community size.
    :ivar sum: Numpy array representing the sum of fixed genes for specified objectives.
    :ivar frontier: Numpy array representing a solution frontier, initialized to zeros.
    :ivar seed: Seed used for initializing the random number generator.
    :ivar rand: Random number generator initialized with the provided seed.
    """

    def __init__(self, frame_path: str | DataFrame, obj_list: List[str], com_size: int = 100,
                 fixed_list: List[str] = None, seed: int = SeedSequence().entropy):
        """Default constructor provides control over default EA parameters.

        Defaults are defined above in the function header.

        Parameters
        -------
        frame_path, string
            Load the frame_path string into a DataFrame.
            Should resolve to a csv file.
            See repository for documentation on what the format of the frame
            should be.
        obj_list, list of strings
            List of column identifiers to use as objectives for the GA.
        com_size, optional, integer
            Size of candidate communities to fix the problem to.
        fixed_list, optional, list
            List of genes to keep fixed in the candidate solutions.
            Should be of the form of the string labels of genes.
        """

        if fixed_list is None:
            fixed_list = []

        self.frame_path = frame_path
        if isinstance(frame_path, DataFrame):
            self.data_frame = frame_path
        else:
            self.data_frame = pd.read_csv(frame_path)
        self.data_numpy = self.data_frame[obj_list].to_numpy()

        if 'GeneName' not in self.data_frame.columns:
            raise AttributeError('Dataset must contain a "GeneName" column')

        self.gene_count = self.data_frame.shape[0]
        self.obj_list = obj_list
        self.fixed_list = fixed_list
        self.fixed_list_nums = np.array(self.data_frame[self.data_frame['GeneName'].isin(fixed_list)].index.to_list())

        # Reorder the dataframe so that the fixed genes appear first
        # to_appear_first = self.fixed_list_ids
        # new_index_order = [*to_appear_first, *self.data_frame.index.difference(to_appear_first)]
        # self.data_frame = self.data_frame.loc[new_index_order].reset_index(drop=True)
        # self.fixed_list_ids = [x for x in range(len(self.fixed_list))]

        self.com_size = com_size - len(fixed_list)
        self.indiv_len = np.arange(self.com_size)
        self.sum = self.data_frame[self.data_frame["GeneName"].isin(self.fixed_list)][
            self.obj_list].sum().to_numpy()
        self.frontier = np.zeros(shape=self.gene_count, dtype=np.int64)  # [0 for x in range(self.gene_count)]
        self.seed = seed
        self.rand = np.random.default_rng(PCG64DXSM(seed))

    def __str__(self):
        """Return all parameters as a formatted string."""
        return (
            f"\tFrame path: {self.frame_path}\n"
            f"\tGene Count: {self.gene_count}\n"
            f"\tObjectives: {self.obj_list}\n"
            f"\tCommunity size: {self.com_size + len(self.fixed_list)}\n"
            f"\tFixed Genes: {self.fixed_list}\n"
            f"\tSeed: {self.seed}"
        )

from typing import List, Tuple, Any

import pandas as pd

from gadgit import GeneInfo, GAInfo


class GAOutput:
    """
    This class represents the output of a Genetic Algorithm (GA) run, including detailed
    information about the genes involved, the elite members, and their rankings. It
    processes and presents results based on problem-specific definitions and data.

    The class retrieves and processes the results of the GA execution, identifies
    explored and unexplored genes, computes rankings considering fixed genes,
    and allows exporting data to a structured DataFrame. It ensures that the input
    dataset contains required fields for processing and provides a comparison mechanism
    for instances.

    :ivar gene_info: Input data and metadata related to genes used in the GA.
    :ivar ga_info: Metadata and processing details for the GA execution.
    :ivar elite: List of indices representing the elite members of the GA run.
    :ivar extra: Additional keyword arguments supplied to the instance.
    :ivar genes_in_elite: List of gene names corresponding to the elite members.
    :ivar frontier: Counter of how often nodes were explored during the GA execution.
    :ivar missed_nodes: List of gene names that were never explored in the GA.
    :ivar rank_pair: Sorted list of tuples containing gene names and their rankings.
    """

    def __init__(self, gene_info: GeneInfo, ga_info: GAInfo, elite: List[Any], verbose: bool = True, **kwargs):
        """
        Initializes an instance of the class responsible for managing genetic algorithm information
        and supplemental gene data. This constructor sets up elite genes, additional configurations,
        and optional verbosity for logging during initialization.

        :param gene_info: Contains the information related to the genes.
        :param ga_info: Contains the information related to the genetic algorithm.
        :param elite: List of the elite individuals required for the genetic algorithm.
        :param verbose: Flag to determine if verbose output logs should be printed during initialization.
        :param kwargs: Additional keyword arguments for extra configurations.
        """

        self.gene_info = gene_info
        self.ga_info = ga_info
        self.elite = list(elite[0])

        self.extra = kwargs

        self._post_run()
        if verbose:
            print(self)

    def _post_run(self) -> None:
        """
        Executes the final processing steps after data analysis and genetic algorithm execution. This
        method prepares the ranked list of genes, updates the frontier data to reflect the number
        of generations, and identifies nodes that were missed in the analysis.

        :raises AttributeError: If the 'GeneName' column is missing from the dataset.
        """

        if 'GeneName' not in self.gene_info.data_frame.columns:
            raise AttributeError('Dataset must contain a "GeneName" column')

        self.genes_in_elite = [self.gene_info.data_frame.loc[ind, 'GeneName'] for ind in sorted(list(self.elite))]
        self.genes_in_elite.extend(self.gene_info.fixed_list)

        self.frontier = self.gene_info.frontier
        self.frontier[self.gene_info.fixed_list_nums] += self.ga_info.gen
        self.missed_nodes = [self.gene_info.data_frame.loc[ind, 'GeneName']
                             for ind, x in enumerate(self.frontier) if x == 0]

        rank_pair = zip(list(self.gene_info.data_frame['GeneName']), self.frontier)
        rank_pair = sorted(rank_pair, reverse=True, key=lambda y: y[1])
        place_list = []
        current_place = 1
        last_element = rank_pair[0][1]
        for i, element in enumerate(rank_pair):
            if element[1] != last_element:
                current_place = i + 1
            last_element = element[1]
            place_list.append(current_place)
        self.rank_pair: List[Tuple[str, int]] = list(zip([x[0] for x in rank_pair], place_list))

    def __str__(self) -> str:
        return (
            f"Genes in elite: {', '.join(self.genes_in_elite)}\n"
            f"Nodes exploration count:\n"
            f"{self.gene_info.frontier}\n"
            f"Nodes never explored (bad): N = {len(self.missed_nodes)}\n"
            f"{', '.join(self.missed_nodes)}\n"
            f"Gene Info:\n"
            f"{self.gene_info}\n"
            f"GA Info:\n"
            f"{self.ga_info}\n"
            f"Gene rankings including fixed genes:\n"
            f"{self.rank_pair}\n"
        )

    def __eq__(self, other) -> bool:
        return (self.rank_pair == other.rank_pair and
                self.missed_nodes == other.missed_nodes and
                self.frontier == other.frontier and
                self.elite == other.elite and
                self.genes_in_elite == other.genes_in_elite)

    def to_df(self) -> pd.DataFrame:
        output = pd.DataFrame([*self.rank_pair]).set_index(0).rename(columns={1: "Rank"})
        genes = [gene[0] for gene in self.rank_pair]
        missed_nodes_values = [True if gene in self.missed_nodes else False for gene in genes]
        missed_nodes_df = pd.DataFrame([genes, missed_nodes_values]).T.set_index(0)
        elite_nodes_values = [True if gene in self.genes_in_elite else False for gene in genes]
        elite_nodes_df = pd.DataFrame([genes, elite_nodes_values]).T.set_index(0)
        fixed_nodes = [True if gene in self.gene_info.fixed_list else False for gene in genes]
        fixed_nodes_df = pd.DataFrame([genes, fixed_nodes]).T.set_index(0)
        output = output.join(missed_nodes_df.rename(columns={1: "Missed"}))
        output = output.join(elite_nodes_df.rename(columns={1: "Elite"}))
        output = output.join(fixed_nodes_df.rename(columns={1: "Fixed"}))
        output.reset_index(inplace=True)
        output.rename(columns={0: "Gene"}, inplace=True)

        return output

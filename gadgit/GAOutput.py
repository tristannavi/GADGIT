from typing import List, Tuple, Any

import pandas as pd

from gadgit import GeneInfo, GAInfo


class GAOutput:
    def __init__(self, gene_info: GeneInfo, ga_info: GAInfo, elite: List[Any], **kwargs):

        self.gene_info = gene_info
        self.ga_info = ga_info
        self.elite = elite[0]

        self.extra = kwargs

        self.__post_run()
        if kwargs.setdefault("print", True):
            print(self)

    def __post_run(self):
        """Takes in the results of some GA and displays information based
           on the problem definition.

           Must contain GeneName in dataframe.

           Parameters
           -------
           gene_info: GeneInfo class
           ga_info: GAInfo class
           hof: list
           """

        if 'GeneName' not in self.gene_info.data_frame.columns:
            raise AttributeError('Dataset must contain a "GeneName" column')

        self.genes_in_elite = [self.gene_info.data_frame.loc[ind, 'GeneName'] for ind in sorted(list(self.elite))]
        self.genes_in_elite.extend(self.gene_info.fixed_list)

        self.frontier = self.gene_info.frontier
        self.frontier[self.gene_info.fixed_list_ids] += self.ga_info.gen
        self.missed_nodes = [self.gene_info.data_frame.loc[ind, 'GeneName']
                             for ind, x in enumerate(self.gene_info.frontier) if x == 0]

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

    def __str__(self):
        return (
            f'Size: {len(self.elite)}',
            f'Genes in elite: {",".join(self.genes_in_elite)}',
            'Nodes exploration count:',
            self.gene_info.frontier,
            f'Nodes never explored (bad): N = {len(self.missed_nodes)}',
            ', '.join(self.missed_nodes),
            'Gene Info:',
            self.gene_info,
            'GA Info',
            self.ga_info,
            'Gene rankings including fixed genes:',
            self.rank_pair
        )

    def __eq__(self, other) -> bool:
        return (self.rank_pair == other.rank_pair and
                self.missed_nodes == other.missed_nodes and
                self.frontier == other.frontier and
                self.elite == other.elite and
                self.genes_in_elite == other.genes_in_elite)

    def to_df(self):
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

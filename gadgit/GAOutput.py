from gadgit import GeneInfo, GAInfo


class GAOutput:
    def __init__(self, gene_info: GeneInfo, ga_info: GAInfo, hof):
        """Takes in the results of some GA and displays information based
           on the problem definition.

           Must contain GeneName in dataframe.

           Parameters
           -------
           gene_info: GeneInfo class
           ga_info: GAInfo class
           hof: DEAP hall of fame object
           """

        self.__post_run(gene_info, ga_info, hof)

    def __post_run(self, gene_info: GeneInfo, ga_info: GAInfo, hof):
        """Takes in the results of some GA and displays information based
           on the problem definition.

           Must contain GeneName in dataframe.

           Parameters
           -------
           gene_info: GeneInfo class
           ga_info: GAInfo class
           hof: DEAP hall of fame object
           """

        if 'GeneName' not in gene_info.data_frame.columns:
            raise AttributeError('GeneNames column not found for post '
                                 'processing script.')

        self.elite = hof[0]
        print('Size: ', len(self.elite))
        buf = 'Index in elite: '
        for ind in sorted(list(self.elite)):
            buf += str(ind) + ', '
        print('Genes in elite: ')
        self.buf = [gene_info.data_frame.loc[ind, 'GeneName'] for ind in sorted(list(self.elite))]
        print('Genes in elite: ', ",".join(self.buf))
        print('Nodes exloration count: ')
        print(gene_info.frontier)
        self.frontier = gene_info.frontier
        self.missed_nodes = [gene_info.data_frame.loc[ind, 'GeneName']
                             for ind, x in enumerate(gene_info.frontier) if x == 0]
        print('Nodes never explored (bad): N =', len(self.missed_nodes))
        print(', '.join(self.missed_nodes))
        print('Gene Info:')
        print(gene_info)
        print('GA Info')
        print(ga_info)
        print('Gene rankings including fixed genes:')
        rank_pair = zip(list(gene_info.data_frame['GeneName']), gene_info.frontier)
        rank_pair = sorted(rank_pair, reverse=True, key=lambda y: y[1])
        place_list = []
        current_place = 1
        last_element = rank_pair[0][1]
        for i, element in enumerate(rank_pair):
            if element[1] != last_element:
                current_place = i + 1
            last_element = element[1]
            place_list.append(current_place)
        self.rank_pair = list(zip([x[0] for x in rank_pair], place_list))
        print(self.rank_pair)

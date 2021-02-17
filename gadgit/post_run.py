def post_run(gene_info, ga_info, pop, stats, hof):
    """Takes in the results of some GA and displays information based on the problem definition.

    Parameters
    -------
    gene_info: GeneInfo class
    ga_info: GAInfo class
    pop: DEAP poplation from a succesful run
    stats: DEAP stats object
    hof: DEAP hall of fame object
    """

    elite = hof[0]
    print('Size: ', len(elite))
    buf = 'Index in elite: '
    for ind in sorted(list(elite)):
        buf += str(ind) + ', '
    print(buf)
    buf = 'Genes in elite: '
    for ind in sorted(list(elite)):
        buf += gene_info.data_frame.iloc[ind]['GeneName'] + ', '
    print(buf)
    print('Nodes exloration count: ')
    print(gene_info.frontier)
    missed_nodes = [gene_info.data_frame.iloc[ind]['GeneName'] for ind, x in enumerate(gene_info.frontier) if x == 0]
    print('Nodes never explored (bad): N =', len(missed_nodes))
    print(', '.join(missed_nodes))
    print('Gene Info:')
    print(gene_info)
    print('GA Info')
    print(ga_info)

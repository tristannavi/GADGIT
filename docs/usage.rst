=====
Usage
=====

To use GADGIT in a project::

    import gadgit

    ga_info = gadgit.GAInfo()

    fixed_genes = ['BRCA1', 'AR', 'ATM', 'CHEK2', 'BRCA2', 'STK11', 'RAD51', 'PTEN', 'BARD1', 'TP53', 'RB1CC1', 'NCOA3', 'PIK3CA', 'PPM1D', 'CASP8']
    gene_info = gadgit.GeneInfo('brca.pkl', ['Betweenness'], fixed_list=fixed_genes)

    pop, stats, hof = gadgit.ga_single(gene_info, ga_info)
    gadgit.post_run(gene_info, ga_info, pop, stats, hof)


An example where the Genetic Algorithm parameters have been customized::

    import gadgit

    ga_info = gadgit.GAInfo(generation=500, population=100, cross_chance=0.5, mut_chance=0.5, tourn_k=5, cross_meth='ops')

    fixed_genes = ['BRCA1', 'AR', 'ATM', 'CHEK2', 'BRCA2', 'STK11', 'RAD51', 'PTEN', 'BARD1', 'TP53', 'RB1CC1', 'NCOA3', 'PIK3CA', 'PPM1D', 'CASP8']
    gene_info = gadgit.GeneInfo('brca.pkl', ['Betweenness'], fixed_list=fixed_genes)

    pop, stats, hof = gadgit.ga_single(gene_info, ga_info)
    gadgit.post_run(gene_info, ga_info, pop, stats, hof)

For more documentation on how to customize the parameter classes see `here <https://gadgit.readthedocs.io/en/latest/installation.html>`_.

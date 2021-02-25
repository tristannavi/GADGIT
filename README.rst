======
GADGIT
======


.. image:: https://img.shields.io/pypi/v/gadgit.svg
        :target: https://pypi.python.org/pypi/gadgit

.. image:: https://img.shields.io/travis/Andesha/gadgit.svg
        :target: https://travis-ci.com/Andesha/gadgit

.. image:: https://readthedocs.org/projects/gadgit/badge/?version=latest
        :target: https://gadgit.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Genetic Algorithm for Disease Gene Identification Toolbox


* Free software: MIT license
* Documentation: https://gadgit.readthedocs.io.


Installation
--------

:code:`pip install gadgit`

Usage
--------

.. code-block:: python

        import gadgit

        ga_info = gadgit.GAInfo()

        fixed_genes = ['BRCA1', 'AR', 'ATM', 'CHEK2', 'BRCA2', 'STK11', 'RAD51', 'PTEN', 'BARD1', 'TP53', 'RB1CC1', 'NCOA3', 'PIK3CA', 'PPM1D', 'CASP8']
        gene_info = gadgit.GeneInfo('brca.pkl', ['Betweenness'], fixed_list=fixed_genes)
        
        pop, stats, hof = gadgit.ga_single(gene_info, ga_info)
        gadgit.post_run(gene_info, ga_info, pop, stats, hof)

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

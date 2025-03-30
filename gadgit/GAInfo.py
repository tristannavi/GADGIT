import numpy as np


class GAInfo:
    """This class stores information regarding evolutionary parameters."""

    def __init__(self, generation=100, population=25, cross_chance=0.75,
                 mut_chance=0.25, tourn_k=3, cross_meth='sdb',
                 seed=np.random.random()):
        """Default constructor provides control over default EA parameters.

        See defaults inside of function header.

        Parameters
        -------
        generation, integer
            Generation count
        population, integer
            Population count
        cross_chance, float/double
            Crossover chance
        mut_chance, float/double
            Mutation chance
        tourn_k, integer
            Tournament selection size
        cross_meth, string
            Crossover method. One of: sdb, ops
                sdb: Safe dealer based
                ops: One point safe
        seed,
            object passed to random.seed function to set seed
        """

        self.gen = generation
        self.pop = population
        self.cxpb = cross_chance
        self.mutpb = mut_chance
        self.nk = tourn_k
        self.cross_meth = cross_meth
        self.seed = seed

    def __str__(self):
        """Return params as string."""

        return (
            f"\tPopulation: {self.gen}\n"
            f"\tGeneration: {self.pop}\n"
            f"\tCrossover: {self.cxpb}\n"
            f"\tMutation: {self.mutpb}\n"
            f"\tTournament: {self.nk}\n"
            f"\tCross Method: {self.cross_meth}\n"
            f"\tSeed: {self.seed}"
        )

class GAInfo:
    """This class stores information regarding evolutionary parameters."""

    def __init__(self, generation=100, population=25, cross_chance=0.75, mut_chance=0.25, tourn_k=3):
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
        """

        self.gen = generation
        self.pop = population
        self.cxpb = cross_chance
        self.mutpb = mut_chance
        self.nk = tourn_k

    def __str__(self):
        """Return params as string."""

        return "Population: {}\nGeneration: {}\nCrossover: {}\nMutation: {}\nTournament: {}".format(self.gen, self.pop, self.cxpb, self.mutpb, self.nk)
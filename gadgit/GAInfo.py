class GAInfo:
    """This class stores information regarding evolutionary parameters."""

    def __init__(self, generation: int = 100, population: int = 25, cross_chance: float = 0.75,
                 mut_chance: float = 0.25, imm_chance: float = 0.60, tourn_k: int = 3, cross_meth: str = 'sdb'):
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
        self.immpb = imm_chance
        self.nk = tourn_k
        self.cross_meth = cross_meth


    def __str__(self) -> str:
        """Return params as string."""

        return (
            f"\tPopulation: {self.pop}\n"
            f"\tGeneration: {self.gen}\n"
            f"\tCrossover: {self.cxpb}\n"
            f"\tMutation: {self.mutpb}\n"
            f"\tTournament: {self.nk}\n"
            f"\tCross Method: {self.cross_meth}\n"
        )

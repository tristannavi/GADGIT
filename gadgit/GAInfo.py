class GAInfo:
    """This class stores information regarding evolutionary parameters."""

    def __init__(self, generation=100, population=25, cross_chance=0.75, mut_chance=0.25, tourn_k=3):
        """Default constructor provides control over default EA parameters."""

        self.gen = generation
        self.pop = population
        self.cxpb = cross_chance
        self.mutpb = mut_chance
        self.nk = tourn_k

    def __str__(self):
        """Return params as string."""

        return "Population: {}\nGeneration: {}\nCrossover: {}\nMutation: {}\nTournament: {}\n".format(self.gen, self.pop, self.cxpb, self.mutpb, self.nk)
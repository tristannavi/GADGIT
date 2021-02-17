import pandas as pd

class GeneInfo:
    """This class stores information regarding a specific problem's biological parameters."""

    def __init__(self, frame_path, obj_list, com_size=100):
        """Default constructor provides control over default EA parameters.

        Defaults are defined above in the function header.

        Parameters
        -------
        frame_path, string
            Load the frame_path string into a DataFrame.
            Should resolve to a pickled pandas DataFrame.
            See repository for documentation on what the format of the frame should be.
        obj_list, list of strings
            List of column identifiers to use as objectives for the GA.
        com_size, integer
            Size of candidate communities to fix the problem to.
        """

        self.frame_path = frame_path
        self.data_frame = pd.read_pickle(frame_path)
        self.gene_count = self.data_frame.shape[0]
        self.obj_list = obj_list
        self.com_size = com_size
        self.frontier = [0 for x in range(self.gene_count)]

    def __str__(self):
        """Return params as string."""

        return "Frame path: {}\nGene Count: {}\nObjectives: {}\nCommunity {}\n".format(self.frame_path, self.gene_count, self.obj_list, self.com_size)

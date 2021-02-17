class GeneInfo:
    """This class stores information regarding a specific problem's biological parameters."""

    def __init__(self, frame_path, obj_list, com_size):
        """Default constructor provides control over default EA parameters.

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

        self.data_frame = pd.read_pickle(frame_path)
        self.gene_count = self.data_frame.shape[0]
        self.com_size = com_size
        self.frontier = [0 for x in range(GENE_COUNT)]

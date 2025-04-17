import pandas as pd


class GeneInfo:
    """This class stores information regarding a specific problem's
    biological parameters."""

    def __init__(self, frame_path, obj_list, com_size=100, fixed_list=None):
        """Default constructor provides control over default EA parameters.

        Defaults are defined above in the function header.

        Parameters
        -------
        frame_path, string
            Load the frame_path string into a DataFrame.
            Should resolve to a csv file.
            See repository for documentation on what the format of the frame
            should be.
        obj_list, list of strings
            List of column identifiers to use as objectives for the GA.
        com_size, optional, integer
            Size of candidate communities to fix the problem to.
        fixed_list, optional, list
            List of genes to keep fixed in the candidate solutions.
            Should be of the form of the string labels of genes.
        """

        if fixed_list is None:
            fixed_list = []

        self.frame_path = frame_path
        self.data_frame = pd.read_csv(frame_path)

        if 'GeneName' not in self.data_frame.columns:
            raise AttributeError('Dataset must contain a "GeneName" column')

        self.gene_count = self.data_frame.shape[0]
        self.obj_list = obj_list
        self.fixed_list = fixed_list
        self.fixed_list_ids = self.data_frame[
            self.data_frame['GeneName'].isin(fixed_list)].index.to_list()

        # Reorder the dataframe so that the fixed genes appear first
        # to_appear_first = self.fixed_list_ids
        # new_index_order = [*to_appear_first, *self.data_frame.index.difference(to_appear_first)]
        # self.data_frame = self.data_frame.loc[new_index_order].reset_index(drop=True)
        # self.fixed_list_ids = [x for x in range(len(self.fixed_list))]

        self.com_size = com_size
        self.frontier = [0 for x in range(self.gene_count)]

    def __str__(self):
        """Return all parameters as a formatted string."""
        return (
            f"\tFrame path: {self.frame_path}\n"
            f"\tGene Count: {self.gene_count}\n"
            f"\tObjectives: {self.obj_list}\n"
            f"\tCommunity size: {self.com_size}\n"
            f"\tFixed Genes: {self.fixed_list}"
        )

"""Kevin Fung - Github Alias: kkf18

WorkFlow user-level parent class.
This module contains the pipeline management of workflow subclasses and manages the flow of manipulated data.

Example:
    flow = WorkFlow(data_dictionary)
    flow.data.clean(datakey)
    flow.nextphase()

Todo:
    * None

"""

from mlflowrate.classes.subclasses.clean import Data
from mlflowrate.classes.subclasses.datasets import DataSets
from mlflowrate.classes.subclasses.explore import DataExplore


class WorkFlow:
    """Pipeline for the management of workflow subclasses.

    Instantiate workflow subclasses and manipulate using these objects.

    Attributes:
        flow_results (dict): Dictionary of different Results objects.
        flow_dfs (dict): Dictionary of different Spark DataFrames containing mixed well data.
        flow_dicts (dict): Dictionary of different organised dictionaries of mixed well data.
        flow_datasets (dict): Dictionary of different Dset objects.

        _track_workflow (dict): Tracks the workflow.
        data (obj): Instantiated Data class.
        datasets (obj): Instantiated DataSets class.
        dataexplore (obj): Instantiated DataExplore class.

    """

    def __init__(self, dfs):
        """Initialise data holders that will contain the final outputs from each class.

        Args:
            dfs (dict): Dictionary of Spark DataFrames containing mixed well features.

        """
        self.flow_results = {}
        self.flow_dfs = {}
        self.flow_dicts = {}
        self.flow_datasets = {}

        self._track_workflow = {"data phase": True, "dataset phase": False, "explore phase": False}
        self.data = Data(dfs)
        self.datasets = None
        self.dataexplore = None

    def status(self):
        """Display the current stage in the data pipeline."""
        print("\nWorkflow")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        for i, (phase, yn) in enumerate(self._track_workflow.items()):
            print("{0}. {1}: {2}".format(i, phase, yn))

    def next_phase(self, repeat_phase=False):
        """Instantiate the next phase of the data pipeline.

        Users must call this function to be able to use the subsequent classes.

        Args:
            repeat_phase (bool):

        """
        if not self._track_workflow["explore phase"]:
            if not self._track_workflow["dataset phase"]:
                if self._track_workflow["data phase"]:
                    self.flow_dfs, self.flow_dicts = self.data.get_data()
                    self.datasets = DataSets(self.flow_dfs, self.flow_dicts)
                    self._track_workflow["dataset phase"] = True
                else:
                    print("No data has been organised!")
                    self._track_workflow["data phase"] = True
            else:
                if repeat_phase:
                    self.flow_dfs, self.flow_dicts = self.data.get_data()
                    self.datasets = DataSets(self.flow_dfs, self.flow_dicts)
                    self._track_workflow["dataset phase"] = True
                    self._track_workflow["explore phase"] = False
                else:
                    self.flow_datasets = self.datasets.get_data()
                    self.dataexplore = DataExplore(self.flow_datasets)
                    self._track_workflow["explore phase"] = True
        else:
            print("\nEnd of exploratory data pipeline reached.")
            self.flow_results = self.dataexplore.get_results()

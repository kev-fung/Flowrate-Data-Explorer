from mlflowrate.data.base import BaseData
from mlflowrate.data.utils.datasets import DataSets
from mlflowrate.data.utils.explore import DataExplore


class WorkFlow:
    """
    make a new flow pipeline object
    """

    def __init__(self, dfs):
        self.flow_results = {}
        self.flow_dfs = {}
        self.flow_dicts = {}
        self.flow_datasets = {}

        self._track_workflow = {"data phase": True, "dataset phase": False, "explore phase": False}
        self.data = BaseData(dfs)
        self.datasets = None
        self.dataexplore = None
        self.explores = []

    def status(self):
        """
        Workflow status
        """
        print("\nWorkflow")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        for i, (phase, yn) in enumerate(self._track_workflow.items()):
            print("{0}. {1}: {2}".format(i, phase, yn))
        if self._track_workflow["explore phase"]:
            print("Explores carried out so far: ")
            for i, exp in enumerate(self.explores):
                print("{0}. {1}".format(i, exp))

    def next_phase(self, explore_name=None):
        """

        :param explore_name:
        """
        if not self._track_workflow["explore phase"]:
            if not self._track_workflow["dataset phase"]:
                if not self._track_workflow["data phase"]:
                    self.flow_dfs, self.flow_dicts = self.data.get_data()
                    self.datasets = DataSets(self.flow_dfs, self.flow_dicts)
                    self._track_workflow["dataset phase"] = True
                else:
                    print("No data has been organised!")
                    self._track_workflow["data phase"] = True
            else:
                self.flow_datasets = self.datasets.get_data()
                self.dataexplore = DataExplore(self.flow_datasets)
                self._track_workflow["explore phase"] = True
        else:
            assert explore_name is not None, "\nProvide an exploration name for the results found in completed phase!"
            assert isinstance(explore_name, str), "\nString name was not passed as argument"
            print("\n{} completed, collecting results.".format(explore_name))
            self.explores.append(explore_name)
            self.flow_results[explore_name] = self.dataexplore.get_results()
            self.dataexplore = DataExplore(self.flow_datasets)
            print("Beginning new exploration")

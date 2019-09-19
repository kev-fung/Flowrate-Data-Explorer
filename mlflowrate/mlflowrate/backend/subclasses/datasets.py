"""Kevin Fung - Github Alias: kkf18

Stage 2: nested class to prepare datasets.
This module contains the features to create new datasets for machine learning.
The idea is to first create your own Spark DataFrames stored in sets, then to finalise the
Spark DataFrame sets into a dset object stored in datasets.

Statistical visualisation tools are provided in aid of feature selection.

dset objects contains the necessary data splits for machine learning e.g. feature set, label set.

Todo:
    * None.

"""

from pyspark.sql import functions as F
from mlflowrate.backend.base import Base
import seaborn as sns
import matplotlib.pyplot as plt


class DataSets(Base):
    """Second stage in data pipeline to assemble datasets for machine learning."""

    def __init__(self, dfs, dicts):
        super().__init__(dfs=dfs, dicts=dicts)

    def make_set(self, setname, align_dates, feats):
        """Make a new Spark DataFrame by picking the relevant features from the different data.

        Users must specify the align_dates argument which makes all selected features from the
        different data to be time consistent with one another. i.e. make everything daily.

        Args:
            setname (str): Name of the new dataset.
            align_dates (str): Options for time consistency: "days", "weeks", "months"
            feats (dict): Dictionary of list of features and the data it orginates from: {data:[feat1, feat2, feat3]}

        """
        set_dict = {}
        for data, feat_list in feats.items():
            for feat in feat_list:
                set_dict[feat] = self.dicts[data][feat].select(
                    F.date_trunc(align_dates, self.dicts[data][feat].datetime).alias("datetime"), "value")

        self.sets[setname] = self._dict2df(set_dict, sort=True, drop_nulls=False)

        # Truncated duplicates can exist in dataframe, so lets drop these samples.
        self.sets[setname] = self.sets[setname].dropDuplicates(["datetime"])

    def status(self):
        """Displays the status of all sets and datasets that have been prepared."""
        print("\n Sets for Preparation")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        for dset, df in self.sets.items():
            print("{0}  |  Samples {1}  |  Is Dataset: {2}".format(dset, df.count(), dset in self.datasets.keys()))

    def distributions(self, setname, grid_size):
        """Plot the kernel density distributions of the Spark DataFrame set.

        Args:
            setname (str): name of the set i.e. created Spark DataFrame.
            grid_size (tuple): size of the grid plot, (width, height)

        """
        assert setname in self.sets.keys(), "{} does not exist in list of sets".format(setname)
        assert len(self.sets[setname].columns) == (
                    grid_size[0] * grid_size[1]) + 1, "grid size must match number of features in set"

        pdset = self.sets[setname].toPandas()
        pdset = pdset.drop(['datetime'], axis=1)
        fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 3, grid_size[0] * 3))
        axs = axs.flatten()

        for c, ax in zip(pdset.columns, axs):
            ax.set_title("{}".format(c))
            sns.distplot(pdset[c], ax=ax, label=c)
            ax.grid(True)

        fig.suptitle('Kernel Density Estimation Distribution')
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)

        display(fig)
        plt.close(fig)

    def pairplot(self, setname):
        """Plot the pairwise bivariate distribution of the specified set.

        Args:
            setname (str): name of the set i.e. created Spark DataFrame.

        """
        assert setname in self.sets.keys(), "{} does not exist in list of sets".format(setname)

        pdset = self.sets[setname].toPandas()
        pdset = pdset.drop(['datetime'], axis=1)

        g = sns.pairplot(pdset)
        fig = g.fig
        fig.suptitle('Pairwise Bivariate Distributions')
        fig.subplots_adjust(top=0.95)

        display(fig)
        plt.close(fig)

    def pearsons(self, setname):
        """Plot Pearson Correlation Matrix of the specified set.

        Args:
            setname (str): name of the set i.e. created Spark DataFrame.

        """
        assert setname in self.sets.keys(), "{} does not exist in list of sets".format(setname)

        pdset = self.sets[setname].toPandas()
        pdset = pdset.drop(['datetime'], axis=1)

        correlations = pdset.corr()

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
        display(fig)
        plt.close(fig)

    def transfer_rows(self, fromset, condition, toset):
        """Based on a Spark Condition, transfer certain rows satisfying this condition from one set to another.

        Args:
            fromset (str): Name of the first set to transfer rows from.
            condition (lambda): Lambda function which contains Spark Where conditional function.
            toset (str): Name of the second set to transfer rows to.

        """
        for feat in self.sets[fromset].columns:
            assert feat in self.sets[toset].columns, "columns from data do not match with columns in set"

        selected_rows = condition(self.sets[fromset])

        order_cols = self.sets[toset].columns
        selected_rows = selected_rows.select(*order_cols)
        self.sets[toset] = self.sets[toset].union(selected_rows).orderBy("datetime")

    def date_range(self, setname, start="1999-01-01", end="2100-01-01"):
        """Threshold the specified set based on a range of dates.

        Args:
            setname (str): Name of the set i.e. created Spark DataFrame.
            start (str): Timestamp of when to start the set: YYYY-MM-DD
            end (str): Timestamp of when to end the set: YYYY-MM-DD

        """
        self.sets[setname] = self.sets[setname].where(
            (self.sets[setname].datetime >= start) & (self.sets[setname].datetime <= end)).orderBy("datetime")

    def make_dataset(self, setname, label=None, feats=None, pandas=False):
        """Finalise sets into dset objects and store into datasets dictionary.

        Args:
            setname (str): Name of the set i.e. created Spark DataFrame.
            label (str): Specify the label column in the set (AKA: y column, prediction variable).
            feats (list): Specify the list of columns that are features in the set (AKA: X columns, predictors).
            pandas (bool): Specify to make dataframe into Pandas format.
        """
        assert setname in self.sets.keys()
        self.datasets[setname] = Dset(self.sets[setname])
        self.datasets[setname].split(label, feats, pandas)

    def cache_data(self, *args):
        """Wrapper method for caching Spark DataFrames to speed things up!

        Args:
            *args: list of arguments specifying which sets to optimise.

        """
        for setname in args:
            assert setname in self.sets.keys(), "Set does not exist in sets"
            self.sets[setname].cache()

    def get_data(self):
        """Returns the dictionary of dset objects"""
        return self.datasets


class Dset:
    """Data storing class for datasets.

    Attributes:
        df (DataFrame): Spark DataFrame containing date, features and labels together.
        is_pandas (bool): Check if DataFrames are in Pandas format or not.
        Xy (DataFrame): Spark or Pandas DataFrame containing features and labels together.
        X (DataFrame): Spark or Pandas DataFrame containing features only.
        y (DataFrame): Spark or Pandas DataFrame containing label only.
        date (DataFrame): Spark or Pandas DataFrame containing date only.

    """

    def __init__(self, df):
        self.df = df.orderBy("datetime")
        self.is_pandas = False
        self.Xy = None
        self.X = None
        self.y = None
        self.date = None

    def split(self, label, features, pandas=False):
        """Method to split the passed in Spark DataFrame into the dataset components.

        Args:
            label (str): Name of the label column in df.
            features (list): List of names of features in df.
            pandas (bool): Option to parse all Spark DataFrames as Pandas instead.

        """
        if not pandas:
            self.Xy = self.df.select(*[feat for feat in self.df.columns if feat not in ["datetime"]])

            if features is None:
                if label is not None:
                    self.X = self.df.select(*[feat for feat in self.df.columns if feat not in ["datetime", label]])
                else:
                    self.X = self.df.select(*[feat for feat in self.df.columns if feat not in ["datetime"]])
            else:
                self.X = self.df.select(*features)

            if label is not None:
                self.y = self.df.select(label)

            self.date = self.df.select("datetime")
        else:
            self.df = self.df.toPandas()
            self.Xy = self.df.drop(['datetime'], axis=1)

            if features is None:
                if label is not None:
                    self.X = self.df.drop(['datetime', label], axis=1)
                else:
                    self.X = self.df.drop(['datetime'], axis=1)
            else:
                self.X = self.df[features]

            if label is not None:
                self.y = self.df[[label]]

            self.date = self.df[['datetime']]
            self.is_pandas = True

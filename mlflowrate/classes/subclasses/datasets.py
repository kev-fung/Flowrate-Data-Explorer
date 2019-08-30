from pyspark.sql import functions as F
from mlflowrate.data.base import BaseData
import seaborn as sns
import matplotlib.pyplot as plt


class DataSets(BaseData):
    """
    Assemble dataframes into train val test sets
    """

    def __init__(self, dfs, dicts):
        super().__init__(dfs=dfs, dicts=dicts)

    def make_set(self, setname, align_dates, feats):
        """

        :param setname:
        :param align_dates:
        :param feats:
        """

        # feats={data:[feat]}
        set_dict = {}
        for data, feat_list in feats.items():
            for feat in feat_list:
                set_dict[feat] = self.dicts[data][feat].select(
                    F.date_trunc(align_dates, self.dicts[data][feat].datetime).alias("datetime"), "value")

        self.sets[setname] = self._dict2df(set_dict, sort=True, drop_nulls=False)

        # Truncated duplicates can exist in dataframe, so lets drop these samples.
        self.sets[setname] = self.sets[setname].dropDuplicates(["datetime"])

    def status(self):
        """

        """
        print("\n Sets for Preparation")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        for dset, df in self.sets.items():
            print("{0}  |  Samples {1}  |  Is Dataset: {2}".format(dset, df.count(), dset in self.datasets.keys()))

    def distributions(self, setname, grid_size):
        """

        :param setname:
        :param grid_size:
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
        """

        :param setname:
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
        """

        :param setname:
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

    def append_rows(self, fromset, condition, toset):
        """

        :param fromset:
        :param condition:
        :param toset:
        """
        for feat in self.sets[fromset].columns:
            assert feat in self.sets[toset].columns, "columns from data do not match with columns in set"

        selected_rows = condition(self.sets[fromset])

        order_cols = self.sets[toset].columns
        selected_rows = selected_rows.select(*order_cols)
        self.sets[toset] = self.sets[toset].union(selected_rows).orderBy("datetime")

    def date_range(self, setname, start, end):
        """
        start = "yyyy-mm-dd"
        :param setname:
        :param start:
        :param end:
        """
        self.sets[setname] = self.sets[setname].where(
            (self.sets[setname].datetime >= start) & (self.sets[setname].datetime <= end)).orderBy("datetime")

    def make_dataset(self, setname, label=None, feats=None, pandas=False):
        """

        :param setname:
        :param label:
        :param feats:
        :param pandas:
        """
        assert setname in self.sets.keys()
        self.datasets[setname] = dset(self.sets[setname])
        self.datasets[setname].split(label, feats, pandas)

    def cache_data(self, *args):
        """

        :param args:
        """
        for setname in args:
            assert setname in self.sets.keys(), "Set does not exist in sets"
            self.sets[setname].cache()

    def get_data(self):
        """

        :return:
        """
        return self.datasets


class dset():
    def __init__(self, df):
        self.df = df.orderBy("datetime")
        self.is_pandas = False
        self.Xy = None
        self.X = None
        self.y = None
        self.date = None

    def split(self, label, features, pandas=False):
        """

        :param label:
        :param features:
        :param pandas:
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

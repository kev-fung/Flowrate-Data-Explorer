"""Kevin Fung - Github Alias: kkf18

Stage 1: nested class to integrate and clean data.
This module contains the integration and cleaning features for messy oil well data.
Imported csv data must be in one of two formats:
    Format1: |datetime|tagname|value|
    Format2: |datetime|feature1|feature2|feature3|

Todo:
    * None.

"""

from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from mlflowrate.backend.base import Base
import re


class Integrate(Base):
    """First of nested backend: integrate raw messy data into consistent, cleaned Spark DataFrames.

    This class provides the tools for cleaning and reorganising data. The user must understand what
    the data consists of and these are provided by the plotting and status features.

    Users must understand that the imported data initially is in a Spark DataFrame, but a corresponding
    dictionary format is created upon instantiation of this class. The dictionary format removes the
    inconsistency sampling of features in Spark DataFrames.

    Attributes:
        _track_org (dict): dictionary of boolean values: if true both Spark DataFrame and Dictionary formats
                            are consistent to each other.
        _formats (list): list of organisational formats

    """

    def __init__(self, dfs):
        super().__init__(dfs=dfs)
        self._track_org = {}
        self._formats = ['date_tag_val_col', 'mult_col', 'dict_col']

        for data in dfs.keys():
            self._track_org[data] = False

    def add_data(self, add_dfs):
        """Add a dictionary of new dfs into the class: {name:df, name:df, name:df etc.}

        Args:
            add_dfs (dict): Dictionary of Spark DataFrames.

        """
        for name, df in add_dfs.items():
            assert name not in self.dfs.keys(), "there is already data with the same name! {}".format(name)
            self.dfs[name] = df
            self._track_org[name] = False

    def status(self, data=None):
        """Display the status of all the data in the current stage or the status of a specific data

        Args:
           data (str): Name of data to check the status of.

        """
        if data is not None:
            assert data in self.dfs.keys()

            print("\n{}".format(data))
            print("~~~~~~~~~~~~~~~~~~")
            print("Organised (dataframe format == dictionary format): {}".format(self._track_org[data]))
            print("Number of samples in dataframe: {}".format(self.dfs[data].count()))
            print("Number of date duplicates in dataframe: {}".format(
                self.dfs[data].count() - self.dfs[data].dropDuplicates(["datetime"]).count()))
            print("Number of samples with null across columns in dictionary:")
            for feat, df in self.dicts[data].items():
                samples = df.count()
                print("{0}  |  Samples {1}  "
                      "|  Nulls {2}  "
                      "|  Duplicates {3}".format(feat, samples, df.where(df["value"].isNull() == True).count(),
                                                 samples - df.dropDuplicates(
                                                    ["datetime"]).count()
                                                 ))
        else:
            print("\nMetadata")
            print("~~~~~~~~~~~~~~~~~~")
            # show all dfs data, all dict data, organised
            print("Name of DataFrame   |   Organised")
            for name, org in self._track_org.items():
                print(" {0}  |  {1} ".format(name, org))

    def show(self, data):
        """Display the data (Spark DataFrame) contained in Integrate.

        Args:
            data (str): Name of the data.
        """
        display(self.dfs[data])

    def set_organised(self, name):
        """Data which is organised will be put into the out_dfs/dicts for the next phase.

        Args:
            name (str): name of data to be set as organised.

        """
        assert name in self.dfs.keys(), "name of data does not exist!"
        self._track_org[name] = True

    def merge_data(self, newname, first, second, axis=0):
        """Append Spark DataFrames vertically or horizontally.

        Args:
            newname (str): new name of the merged DataFrames
            first (str): name of data to join from.
            second (str): name of data to join to.
            axis (int): Direction of append.
        """
        assert axis in [0, 1], "axis must be either 0 or 1"
        assert first in self.dfs.keys(), "name of first data does not exist"
        assert second in self.dfs.keys(), "name of second data does not exist"

        if axis is 0:
            """Bottom append"""
            assert len(self.dfs[first].columns) == len(self.dfs[second].columns), "DataFrames columns do not match"
            self.dfs[newname] = self.dfs[first].union(self.dfs[second])
            self.dicts[newname] = self._df2dict(self.dfs[newname])
            self._track_org[newname] = False
        else:
            """Left join columns"""
            print("To be implemented.")

    def clean_data(self, data, null_col=None, remove_nulls=False, char_col=None, remove_char=None, zeros_col=None,
                   remove_zeros=False, avg_over=None, is_dict=False):
        """Multi-functional method to clean data.

        If column arguments are not specified. Method will assume the entire DataFrame.

        Args:
            data (str): name of the data to clean.
            null_col (str): name of the column in data to remove null from.
            remove_nulls (bool): option to remove nulls.
            char_col (str): name of the column in data to remove char from.
            remove_char (bool): option to remove char
            zeros_col (str): name of the column in data to remove zeros from.
            remove_zeros (bool): option to remove zeros.
            avg_over (str): periodicity for averaging: "day", "week", "month"
            is_dict (bool): special option to select for averaging if Spark DataFrame is inconsistent.

        """
        assert data in self.dfs.keys(), "data must be a spark dataframe stored within dfs"
        assert "datetime" in self.dfs[data].columns, "no datetime columns can be found in dataframe"

        if remove_nulls:
            if null_col is not None:
                assert null_col in self.dfs[data].columns
                self.dfs[data] = self.dfs[data].filter(self.dfs[data][null_col].isNotNull()).orderBy("datetime")
            else:
                self.dfs[data] = self.dfs[data].na.drop().orderBy("datetime")
                self.dicts[data] = self._df2dict(self.dfs[data])

        if remove_zeros:
            assert zeros_col is not None, "Must specify the zeros column"
            self.dfs[data] = self.dfs[data].where(self.dfs[data][zeros_col] != 0.0).orderBy("datetime")

        if remove_char is not None:
            assert char_col is not None, "Must specify the feature column"
            self.dfs[data] = self.dfs[data].where(self.dfs[data][char_col] != remove_char).orderBy("datetime")

        if avg_over is not None:
            if not is_dict:
                self.dfs[data] = self._avg(self.dfs[data], avg_over)
            else:
                for feat, df in self.dicts[data].items():
                    self.dicts[data][feat] = self._avg(df, avg_over)

    def _avg(self, data, avg_over):
        """Average the data over a time period.

        Args:
            data (obj): data to average.
            avg_over (str): periodicity specification.

        Returns:
            Averaged Spark DataFrame

        """
        features = data.columns
        features.remove("datetime")

        new_df = data.withColumn(avg_over, F.date_trunc(avg_over, data["datetime"]))
        new_df = new_df \
            .groupBy(avg_over) \
            .agg(*[F.avg(feat) for feat in features]) \
            .orderBy(avg_over)

        for feat in features:
            new_df = new_df.withColumnRenamed("avg({})".format(feat), feat)

        new_df = new_df.withColumnRenamed(avg_over, "datetime")
        return new_df

    def edit_col(self, data, feature, **kwargs):
        """Multi-functional method to edit a specific column in data.

        Args:
            data (str): name of the data.
            feature (str): name of the specific column in data.
            **kwargs: Options for editing the columns of data: typ, newname, std

        """
        assert data in self.dicts.keys(), "data in dictionary format not found"

        if not bool(kwargs.keys()):
            print("No options were passed in")
            print("To recast col:")
            print("   edit_col(data=str, feature=str, typ=str)")
            print("To rename col:")
            print("   edit_col(data=str, feature=str, newname=str)")
            print("To standardise col:")
            print("   edit_col(data=str, feature=str, std=True)")

        if "typ" in kwargs.keys():
            self.dfs[data] = self.dfs[data].withColumn("_", self.dfs[data][feature].cast(kwargs["typ"]))
            self.dfs[data] = self.dfs[data].drop(self.dfs[data][feature]).withColumnRenamed("_", feature)
            self.dicts[data][feature] = self.dicts[data][feature].withColumn("-",
                                                                             self.dicts[data][feature]["value"].cast(
                                                                                 kwargs["typ"]))
            self.dicts[data][feature] = self.dicts[data][feature].drop("value").withColumnRenamed("_", "value")

        elif "newname" in kwargs.keys():
            self.dfs[data] = self.dfs[data].withColumnRenamed(feature, kwargs["newname"])
            self.dicts[data][kwargs["newname"]] = self.dicts[data][feature]
            self.dicts[data] = {feat: df for feat, df in self.dicts[data].items() if feat not in [feature]}

        elif "std" in kwargs.keys():

            mean = self.dicts[data][feature].select(F.mean("value")).collect()[0][0]
            stddev = self.dicts[data][feature].select(F.stddev("value")).collect()[0][0]

            def std(val):
                return (val - mean) / stddev

            udf = F.udf(std, DoubleType())
            self.dicts[data][feature] = self.dicts[data][feature].withColumn("stdd", udf("value"))
            self.dicts[data][feature] = self.dicts[data][feature].select("datetime",
                                                                         self.dicts[data][feature]["stdd"].alias(
                                                                             "value"))
            self.dfs[data] = self._dict2df(self.dicts[data], sort=True)

        else:
            print("No options were given")

    def drop_col(self, data, *features):
        """Remove a number of columns from Spark DataFrame.

        Args:
            data (str): name of data to drop columns.
            *features: argument list of the features to drop in data.

        """
        assert data in self.dicts.keys(), "data in dictionary format not found"
        self.dfs[data] = self.dfs[data].select(*[feat for feat in self.dfs[data].columns if feat not in features])
        self.dicts[data] = {feat: df for feat, df in self.dicts[data].items() if feat not in features}

    def select_col(self, data, *features):
        """Keep a number of columns from Spark DataFrame to be the Spark DataFrame.

        Args:
            data (str): name of data to keep columns.
            *features: argument list of the features to keep in data.

        """
        assert data in self.dicts.keys(), "data in dictionary format not found"
        self.dfs[data] = self.dfs[data].select("datetime", *features)
        self.dicts[data] = {feat: df for feat, df in self.dicts[data].items() if feat in features}

    def organise_data(self, name, dfmat, **kwargs):
        """Key method for integrating mixed data into consistent Spark DataFrame formats.

        Users are provided three different options to format the disorganised data:
            date_tag_val_col: If the Spark DataFrame has only three columns |date|tagname|value|, then
                              the user is provided options to sort data by distinct features, and further
                              categorically by oil wells. An inconsistent Spark DataFrame and corresponding
                              Dictionary format is produced.
            mult_col: Make a corresponding Dictionary format from a Spark DataFrame.
            dict_col: Make a corresponding and consistent Spark DataFrame from a Dictionary format.

        Args:
            name (str): Name of data to format.
            dfmat (str): Options to format data.
            **kwargs: Further args for sorting date, tagname, and value data: distinct_oilwells, change_sensor_names.

        """
        assert name in self.dfs.keys(), "there is no data with the name {}".format(name)
        assert dfmat in self._formats, "format does not exist! select 'date_tag_val_col', 'mult_col', dict_col"

        if dfmat == 'date_tag_val_col':
            # NOTE: Will only return a perfect df if the data features had exactly the same number of samples

            assert len(self.dfs[name].columns) == 3, "DataFrame does not have correct number of columns!"
            assert "tag" in self.dfs[name].columns, "No 'tag' column"
            assert "datetime" in self.dfs[name].columns, "No 'datetime' column"
            assert "value" in self.dfs[name].columns, "No 'value' column"

            df_dict = {}
            unq_items = self.dfs[name].select("tag").distinct()
            n = unq_items.rdd.map(lambda x: x["tag"]).collect()

            for key in n:
                df_dict[key] = self.dfs[name].orderBy("datetime").where(self.dfs[name]["tag"] == key).select(
                    self.dfs[name]["datetime"], self.dfs[name]["value"])

            # df_dict now has a collection of distinct keys corresponding to unique tag names.
            # tag name contains information of the oilwell, and type of sensor measurement.

            if "distinct_oilwells" in kwargs.keys():
                # we want to get individual dictionaries corresponding to each oilwell!
                # kwarg should be a dict with {nameofoilwell : ["tag_name", "tag_name", ...]}
                for ow, tag_list in kwargs["distinct_oilwells"].items():
                    out_dict = {}
                    regex_list = []
                    for tag_ in tag_list:
                        regex_list.append(re.compile(tag_))

                    for (key, val) in df_dict.items():
                        for reg in regex_list:
                            if re.match(reg, key):
                                out_dict[key] = val

                    self.dicts[ow] = out_dict

                    if "change_sensor_names" in kwargs.keys():
                        self.dicts[ow] = self._change_sensor_names(self.dicts[ow], kwargs["change_sensor_names"])

                    self.dfs[ow] = self._dict2df(self.dicts[ow], sort=True)
                    self._track_org[ow] = False

            else:
                self.dicts[name] = df_dict

                if "change_sensor_names" in kwargs.keys():
                    self.dicts[name] = self._change_sensor_names(self.dicts[name], kwargs["change_sensor_names"])

                self.dfs[name] = self._dict2df(self.dicts[name])
                self._track_org[name] = False

        elif dfmat == 'mult_col':
            self.dfs[name] = self.dfs[name]
            self.dicts[name] = self._df2dict(self.dfs[name])
            self._track_org[name] = False

        elif dfmat == 'dict_col':
            self.dfs[name] = self._dict2df(self.dicts[name], sort=True)
            self.dicts[name] = self._df2dict(self.dfs[name])
            self._track_org[name] = False

        else:
            print("Unusual Error!")

    def get_data(self):
        """Returns the formatted dictionaries of Spark DataFrames and Dictionaries.

        Returns:
            Dictionaries of Spark DataFrames and Dictionaries.

        """
        out_dfs = {}
        out_dicts = {}
        for data, org in self._track_org.items():
            if org:
                out_dfs[data] = self.dfs[data]
                out_dicts[data] = self.dicts[data]

        return out_dfs, out_dicts

    def _change_sensor_names(self, in_dict, decode):
        """Private method to change the tagnames of a Data in Dictionary Format.

        Args:
            in_dict (dict): Data in dictionary format.
            decode (dict): Dictionary whose key value pairs correspond to translating tagnames.

        """
        new_dict = {}
        for key, val in in_dict.items():
            for k, new_key in decode.items():
                regex = re.compile(k)
                if re.match(regex, key):
                    new_dict[new_key] = val

        return new_dict

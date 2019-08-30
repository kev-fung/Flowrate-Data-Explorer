from pyspark.sql import functions as F
from mlflowrate.data.base import BaseData
import re


class DataCleaner(BaseData):
    """
    Store and reorganise messy data into clean formats
    """

    def __init__(self, dfs):
        super().__init__(dfs=dfs)
        self._track_org = {}
        self._formats = ['date_tag_val_col', 'mult_col', 'dict_col']

        for data in dfs.keys():
            self._track_org[data] = False

    def add_data(self, add_dfs):
        """Add a dictionary of new dfs into the class: {name:df, name:df, name:df etc.}
        :param add_dfs:
        """
        for name, df in add_dfs.items():
            assert name not in self.dfs.keys(), "there is already data with the same name! {}".format(name)
            self.dfs[name] = df
            self._track_org[name] = False

    def status(self, data=None):
        """
        Description of dfs
        :param data:
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
                print("{0}  |  Samples {1}  |  Nulls {2}"
                      "  |  Duplicates {3}".format(feat, samples,
                                                   df.where(df["value"].isNull() == True).count(),
                                                   samples - df.dropDuplicates(["datetime"]).count()))
        else:
            print("\nMetadata")
            print("~~~~~~~~~~~~~~~~~~")
            #             show all dfs data, all dict data, organised
            print("Name of DataFrame   |   Organised")
            for name, org in self._track_org.items():
                print(" {0}  |  {1} ".format(name, org))

    def set_organised(self, name):
        """
        Set data which is organised to be tracked and put into the out_dfs/dicts for the next phase
        :param name:
        """

        assert name in self.dfs.keys(), "name of data does not exist!"
        self._track_org[name] = True

    def merge_data(self, newname, first, second, axis=0):
        """
        Merge two given dfs
        :param newname: name of new dfs
        :param first: name of first dfs
        :param second: name of second dfs
        :param axis: axis on which to merge (0: bottom append, 1:left join)
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

    def clean_data(self, data, null_col=None, remove_nulls=False, char_col=None, remove_char=None, avg_over=None,
                   is_dict=False):
        """
        Helper function to clean dirty and organised dfs
        :param data: the dfs to be cleaned
        :param null_col: selected column to remove null values
        :param remove_nulls: null remover flag
        :param char_col:
        :param remove_char:
        :param avg_over:
        :param is_dict:
        """
        assert data in self.dfs.keys(), "data must be a spark dataframe stored within dfs"
        assert "datetime" in self.dfs[data].columns, "no datetime columns can be found in dataframe"

        if remove_nulls:
            if null_col is not None:
                assert null_col in self.dfs[data].columns
                self.dfs[data] = self.dfs[data].filter(self.dfs[data][null_col].isNotNull())
            else:
                self.dfs[data] = self.dfs[data].na.drop()
                self.dicts[data] = self._df2dict(self.dfs[data])

        if remove_char is not None:
            assert char_col is not None, "Must specify the feature column"
            self.dfs[data] = self.dfs[data].where(self.dfs[data][char_col] != remove_char)

        if avg_over is not None:
            if not is_dict:
                self.dfs[data] = self._avg(self.dfs[data], avg_over)
            else:
                for feat, df in self.dicts[data].items():
                    self.dicts[data][feat] = self._avg(df, avg_over)

    def _avg(self, data, avg_over):
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
        """
        Edit feature columns for a particular dfs. Currently two options are available
            casting (typ): cast a column dtype value to a different type
            renaming (newname): rename the column name
        :param data: the dfs to edit
        :param feature: the selected feature column
        :param kwargs: keyword arguments for editing:
                        typ: new column type
                        newname: new column name
        """
        assert data in self.dicts.keys(), "data in dictionary format not found"
        if not bool(kwargs.keys()):
            print("No options were passed in")
            print("To recast col:")
            print("   edit_col(data=str, feature=str, typ=str)")
            print("To rename col:")
            print("   edit_col(data=str, feature=str, newname=str)")
            print("See docstrings for more information.")
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
        else:
            print("No options were given")

    def drop_col(self, data, *features):
        """
        Drop one/more columns from a given dfs
        :param data:
        :param features:
        """
        assert data in self.dicts.keys(), "data in dictionary format not found"
        self.dfs[data] = self.dfs[data].select(*[feat for feat in self.dfs[data].columns if feat not in features])
        self.dicts[data] = {feat: df for feat, df in self.dicts[data].items() if feat not in features}

    def select_col(self, data, *features):
        """
        Select one/more columns from a given dfs
        :param data:
        :param features:
        """
        assert data in self.dicts.keys(), "data in dictionary format not found"
        self.dfs[data] = self.dfs[data].select("datetime", *features)
        self.dicts[data] = {feat: df for feat, df in self.dicts[data].items() if feat in features}

    def organise_data(self, name, dfmat, **kwargs):
        """

        :param name:
        :param dfmat:
        :param kwargs:
        """
        assert name in self.dfs.keys(), "there is no data with the name {}".format(name)
        assert dfmat in self._formats, "format does not exist! select 'date_tag_val_col', 'mult_col', dict_col"

        if dfmat == 'date_tag_val_col':  # Will only return a perfect df if the data features
            # had exactly the same number of samples
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

            #           df_dict now has a collection of distinct keys corresponding to unique tag names.
            #           tag name contains information of the oilwell, and type of sensor measurement.

            if "distinct_oilwells" in kwargs.keys():
                #           we want to get individual dictionaries corresponding to each oilwell!
                #           kwarg should be a dict with {nameofoilwell : ["tag_name", "tag_name", ...]}
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
        """
        Get dfs/dicts of data which have been organised
        :return:
        """
        out_dfs = {}
        out_dicts = {}
        for data, org in self._track_org.items():
            if org:
                out_dfs[data] = self.dfs[data]
                out_dicts[data] = self.dicts[data]

        return out_dfs, out_dicts

    def _change_sensor_names(self, in_dict, decode):
        """

        :param in_dict:
        :param decode:
        :return:
        """
        new_dict = {}
        for key, val in in_dict.items():
            for k, new_key in decode.items():
                regex = re.compile(k)
                if re.match(regex, key):
                    new_dict[new_key] = val

        return new_dict

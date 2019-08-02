"""Subclass of Data for Cataloguing Data

This module contains two classes which would organise any imported
spark dataframe into dictionaries or certain dataframe structures
in the Azure Databricks environment.

Examples:
    A dataframe with mixed label names can be separated out and
    sorted into a dictionary containing the corresponding
    samples.

    A dictionary of dataframes can be converted into a larger
    dataframe by joining the dictionary's dataframes.

@author: Kevin Fung
"""
import re
import os
import sys
import importlib.util
from pathlib import Path


def import_mod(module_name):
    """Method to be able to import homemade .py modules in Azure Databricks
        This must be declared and called before importing any homemade modules!

        Args:
            module_name (str): Name of the module to import

    """
    cwd = os.getcwd()
    my_git_repo_exists = Path('{}/acse-9-independent-research-project-kkf18'.format(cwd))
    spec = importlib.util.spec_from_file_location("{}.py".format(module_name),
                                                  "{}/{}.py".format(my_git_repo_exists, module_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # load module into the sys module dictionary so it can be imported in
    sys.modules[module_name] = module
    assert module_name in sys.modules.keys()
    print("Import successful")


# Import homemade modules
import_mod("Data")
import Data


class DataframeTools(Data.Data):
    """Parent class for manipulating spark dataframes"""

    def __init__(self, df):
        super().__init__(df)

    def append_data(self, old_df, new_df, verbatim=True):
        """Append another dataframe below the current one.
            New dataframe must have the same columns as the original dataframe.

        Args:
            old_df (dataframe): Old dataframe to append new one to
            new_df (dataframe): New dataframe
            verbatim (bool): Print counts before and after appending

        Returns:
            df (dataframe): joined dataframe

        """
        if verbatim is True:
            print("\nCurrent samples: ", old_df.count())
            print("Appending samples: ", new_df.count())
            df = old_df.union(new_df)
            print("Joined samples: ", df.count())
        else:
            df = old_df.union(new_df)

        return df

    def is_null(self, df):
        """Check all columns in dataframe for null values.

        Args:
          df (dataframe): Input dataframe

        """
        print("\nNumber of samples with null across columns:")
        for col in df.schema:
            head = col.name
            print(head, df.where(df[head].isNull() == True).count())

    def null2zero(self, head, df):
        """Change null values to zero in column

        Args:
            head (str): Column name to replace nulls
            df (dataframe): Input dataframe

        Returns:
            dfs (dataframe): dataframe with replaced null values

        """
        print("\nReplacing null values with zero")
        dfs = df.na.fill(0, head)
        return dfs

    def removenulls(self, head, df):
        """Remove any null values in column

        Args:
            head (str): Column name to remove nulls
            df (dataframe): Input dataframe

        Returns:
            dfs (dataframe): dataframe with null values removed in named column

        """
        dfs = df.filter(df[head].isNotNull())
        return dfs

    def df2dict(self, df, date_head="datetime"):
        """Convert a dataframe of collected timeseries features into an organised dictionary.
            ow_dict = {feature name : timeseries dataframe}

        Args:
            df (dataframe): input dataframe with different timeseries features
            date_head (str): column name of datetime in df

        Returns:
            ow_dict (dict): organised dictionary of features from input dataframe

        """
        assert date_head in df.schema.names, "no date column in given dataframe!"

        ow_dict = {}
        for head in df.schema.names:
            if head == date_head:
                continue
            ow_dict[head] = df.select(df[date_head].alias("datetime"), df[head].alias("value"))
        return ow_dict

    def dict2df(self, ow_dict):
        """Convert a dictionary of timeseries features into a single dataframe of collected features.

        Args:
            ow_dict (dict): EXAMPLE K:V FORMAT {feature name : timeseries dataframe}

        Returns:
            dfs (dataframe): dataframe of combined features with FORMAT: datetime|feature1|feature2|etc.

        """
        dfs = ow_dict[list(ow_dict.keys())[0]]
        dfs = dfs.select(dfs["datetime"], dfs["value"].alias(list(ow_dict.keys())[0]))

        for i, (head, df) in enumerate(ow_dict.items()):
            if i == 0:
                continue
            df = df.select(df["datetime"], df["value"].alias(head))
            dfs = dfs.join(df, df.datetime == dfs.datetime, how="left").drop(df.datetime)

        return dfs


class DictionaryTools(DataframeTools):
    """Subclass of Dataframe tools to reorganise dataframes into dictionaries.
      Tools for visualisation and preprocessing included.
    """

    def __init__(self, df):
        super().__init__(df)
        # if df_dict is None:
        #     self.df_dict = {}
        # else:
        #     self.df_dict = df_dict
        # self.headers = [h.name for h in df.schema]

    def separate2dict(self, df, label_head, x_head='datetime', y_head='value'):
        """Collect rows which contain the same label in label_head column and
            put them as a key value pair in dictionary. Applicable for messy timeseries data.

        Args:
            df (dataframe): input dataframe
            label_head (str): column to separate the data
            x_head (str): first column (normally time) to reconstruct corresponding dataframe
            y_head (str): second column (normally value) to reconstruct corresponding dataframe

        """
        assert label_head in df.schema.names, "Header does not exist in dataframe!"

        df_dict = {}
        unq_items = df.select(label_head).distinct()
        n = unq_items.rdd.map(lambda x: x[label_head]).collect()
        for key in n:
            df_dict[key] = df.orderBy(x_head).where(df[label_head] == key).select(df[x_head], df[y_head])

        return df_dict

    def separate2dict_substr(self, df_dict, regex):
        """Separate data out even further based on distinct substrings in target label column.
            Only for dictionaries with split data already. I.e. Further separates the data into
            finer dictionaries based on given distinct substrings.

        Args:
            df_dict (dict): Sorted timeseries data which has already been separated
            regex (str): substring to match inside label, FORMAT: RE Wildcard

        Returns:
            out_dict (dict): Dictionary with only key value pairs from old dictionary that matched
                                with the regex.

        """
        out_dict = {}
        for (key, val) in df_dict.items():
            for reg in regex:
                if re.match(reg, key):
                    out_dict[key] = val

        return out_dict

    def decode_keys(self, in_dict, decode_dict):
        """Replace old keys with new key names given a decode_dict dictionary.

          Args:
              in_dict (dict): input dictionary
              decode_dict (dict): dictionary of old keys with new key names

          Returns:
              new_dict (dict): dictionary with updated keys

        """
        new_dict = {}
        for key, val in in_dict.items():
            for k, new_key in decode_dict.items():
                regex = re.compile(k)
                if re.match(regex, key):
                    # print("Replacing ", key, ", with ", new_key)
                    new_dict[new_key] = val
                    # for i, j in new_dict.items():
                    # print(i, " : ", j)

        return new_dict

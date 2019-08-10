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
    """Method to be able to import homemade .py modules in Azure Databricks. This must be declared and called before
    importing any homemade modules!
    Args:
        module_name (str): Name of the module to import

    Returns:
        None
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
from Data import plot_ts, add_year_col, add_quart_col


class DataframeTools:
    """Parent class for manipulating spark dataframes"""

    def append_data(self, old_df, new_df, v=True):
        """Append another Spark DataFrame below the current one. New DataFrame must have the same columns as
        the original DataFrame.
        Args:
            old_df (Spark DataFrame): Old DataFrame to append new one to
            new_df (Spark DataFrame): New DataFrame
            v (bool): print verbatim

        Returns:
            Spark DataFrame: joined Spark DataFrame
        """
        if v is True:
            print("\nCurrent samples: ", old_df.count())
            print("Appending samples: ", new_df.count())
            df = old_df.union(new_df)
            print("Joined samples: ", df.count())
        else:
            df = old_df.union(new_df)

        return df

    def is_null(self, df):
        """Check all columns for null values in Spark DataFrame
        Args:
          df (Spark DataFrame): Input Spark DataFrame

        Returns:
            None
        """
        print("\nNumber of samples with null across columns:")
        for col in df.schema:
            head = col.name
            print(head, df.where(df[head].isNull() == True).count())

    def null2zero(self, df, head):
        """Change null values to zero in column of Spark DataFrame
        Args:
            df (Spark DataFrame): Input Spark DataFrame
            head (str): Column name to replace nulls

        Returns:
            Spark DataFrame: Spark DataFrame with replaced null values
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
        """Convert a Spark DataFrame of columns of timeseries features into an organised dictionary.
            EXAMPLE:
                ow_dict = {feature name : timeseries dataframe}
        Args:
            df (Spark DataFrame): input Spark DataFrame with different features
            date_head (str): column name of datetime in Spark DataFrame

        Returns:
            Dict: organised dictionary of features from Spark DataFrame
        """
        assert date_head in df.schema.names, "no date column in given DataFrame!"

        ow_dict = {}
        for head in df.schema.names:
            if head == date_head:
                continue
            ow_dict[head] = df.select(df[date_head].alias("datetime"), df[head].alias("value"))
        return ow_dict

    def dict2df(self, ow_dict, datename="datetime", valname="value"):
        """Convert a dictionary of timeseries features into a Spark DataFrame of columns of these features.
        Each DataFrame in the dictionary must have the same formatted datetime columns and sizes!
        Args:
            ow_dict (dict): Input dictionary of features
            datename (str): Name of the column to join by
            valname (str): Name of the value column

        Returns:
            Spark DataFrame: DataFrame with columns of features
        """
        for df in ow_dict.values():
            assert datename in df.schema.names, "Inconsistent datetime column names"
            assert valname in df.schema.names, "Inconsistent value column names"

        # Pull the first kv pair in dictionary out as a DataFrame
        dfs = ow_dict[list(ow_dict.keys())[0]]
        dfs = dfs.select(dfs[datename], dfs[valname].alias(list(ow_dict.keys())[0]))

        for i, (head, df) in enumerate(ow_dict.items()):
            if i == 0:
                continue
            df = df.select(df[datename], df[valname].alias(head))
            dfs = dfs.join(df, df[datename] == dfs[datename], how="left").drop(df[datename])

        return dfs


class DictionaryTools(DataframeTools):
    """Subclass of DataFrame tools to reorganise DataFrames into dictionaries."""

    def separate2dict(self, df, label_head, x_head='datetime', y_head='value'):
        """Collect rows which contain the same label in label_head column and put them as a key value pair in
        dictionary. Applicable for messy timeseries data.
        Args:
            df (Spark DataFrame): input Spark DataFrame
            label_head (str): column to separate the data
            x_head (str): first column (normally time) to reconstruct corresponding DataFrame
            y_head (str): second column (normally value) to reconstruct corresponding DataFrame

        Returns:
            Dictionary: Dictionary of Spark DataFrames splitted by disctinct values in label_head column
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
            df_dict (dict): sorted timeseries data which has already been separated
            regex (str): substring to match inside label, FORMAT: RE Wildcard

        Returns:
            Dictionary: Dictionary with only key value pairs from old dictionary that matched with the regex.
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
              Dictionary: dictionary with updated keys
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

import unittest
from pandas.util.testing import assert_frame_equal
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions
spark = SparkSession.builder.getOrCreate()


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
from Data import Data


class TestData(unittest.TestCase):
    def test_add_year_col(self):
        """(TRIVIAL test) Test if year column is appended"""

        # Contruct dummy dataframe with datetime
        date, years, months = [], 3, 5
        year_res = []
        for year in range(1, years):
            for mon in range(1, months):
                date.append("201{}-0{}-01 00:00:00".format(year, mon))
                year_res.append(int("201{}".format(year)))
        value = [i for i in range(years * months)]
        df = [(i, j) for i, j in zip(date, value)]
        df1 = spark.createDataFrame(df, ["datetime", "value"])
        df1 = df1.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), df1["value"])

        # Construct expected resulting dataframe
        test1 = [(i, j, k) for i, j, k in zip(date, value, year_res)]
        test = spark.createDataFrame(test1, ["datetime", "value", "year"])
        test = test.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), test["value"],
                           test["year"])
        test = test.toPandas().astype({'year': 'int32'})

        # Run method
        d = Data(df1)
        result = d.add_year_col(df1)
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)

    def test_add_quart_col(self):
        """(TRIVIAL test) Test if quarter column is appended"""

        # Contruct dummy dataframe with datetime
        date, years, months = [], 2, 12
        quart_res = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
        for year in range(1, years):
            for mon in range(1, months):
                date.append("201{}-0{}-01 00:00:00".format(year, mon))
        value = [i for i in range(years * months)]
        df = [(i, j) for i, j in zip(date, value)]
        df1 = spark.createDataFrame(df, ["datetime", "value"])
        df1 = df1.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), df1["value"])

        # Construct expected resulting dataframe
        test1 = [(i, j, k) for i, j, k in zip(date, value, quart_res)]
        test = spark.createDataFrame(test1, ["datetime", "value", "quarter"])
        test = test.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), test["value"],
                           test["quarter"])
        test = test.toPandas().astype({'quarter': 'int32'})

        # Run method
        d = Data(df1)
        result = d.add_quart_col(df1)
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)


if __name__ == '__main__':
    unittest.main()

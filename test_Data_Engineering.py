import unittest
from pandas.util.testing import assert_frame_equal

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions
spark = SparkSession.builder.getOrCreate()

# System imports
import os
import sys
import importlib.util
from pathlib import Path

# import method for databricks cluster
def import_mod(module_name):
    cwd = os.getcwd()
    my_git_repo_exists = Path('{}/acse-9-independent-research-project-kkf18'.format(cwd))

    spec = importlib.util.spec_from_file_location("{}.py".format(module_name),
                                                  "{}/{}.py".format(my_git_repo_exists, module_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # load module into the sys module dictionary so it can be imported in
    sys.modules[module_name] = module

    print("Import successful")

    assert module_name in sys.modules.keys()

import_mod("Data_Engineering")
import Data_Engineering as det


class TestAppendData(unittest.TestCase):
    def test_append(self):
        """
        Test that method attaches dataframe to the bottom of the given dataframe
        """
        # Construct dummy dataframes
        dummy1 = [("John", 1.0, 2, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, 6)]
        df1 = spark.createDataFrame(dummy1, ["name", "a", "b", "c", "d", "e"])
        dummy2 = [("JJ", 1.0, 2, 3, 8, 5), ("Bizarre", 1.8, 3, 3, 5, 6)]
        df2 = spark.createDataFrame(dummy2, ["name", "a", "b", "c", "d", "e"])

        # Construct expected dataframe
        test1 = [("John", 1.0, 2, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, 6), ("JJ", 1.0, 2, 3, 8, 5), ("Bizarre", 1.8, 3, 3, 5, 6)]
        test = spark.createDataFrame(test1, ["name", "a", "b", "c", "d", "e"])
        test = test.toPandas()

        # Run method
        dataeng = det.DataframeTools(df1)
        dataeng.append_data(df2)
        result = dataeng.df.toPandas()

        # Test equality
        assert_frame_equal(result, test)

class Testnull2zero(unittest.TestCase):
    def test_null2zero:
        """
        Test if null function works
        """
        # Construct dummy dataframe
        dummy1 = [("John", 1.0, None, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, None)]
        df1 = spark.createDataFrame(dummy1, ["name", "a", "b", "c", "d", "e"])

        # Construct test dataframe
        test1 = [("John", 1.0, 0, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, 0)]
        test = spark.createDataFrame(test1, ["name", "a", "b", "c", "d", "e"])
        test = test.toPandas()

        # Run method
        dataeng = det.DataframeTools(df1)
        result = dataeng.null2zero("b", df1)
        result = dataeng.null2zero("e", result)
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)

class Testavgperiodday(unittest.TestCase):
    def test_avg_over_period(self):
        """
        Test if averaging works
        """

        # Contruct dummy datetime|value dataframe
        date, days = [], 5
        for day in range(1, days+1):
            for hour in range(24):
                date.append("2019-07-0{} {}:00:00".format(day, hour))

        value = [i for i in range(days*24)]
        df = [(i,j) for i,j in zip(date, value)]
        df1 = spark.createDataFrame(df, ["datetime", "value"])
        df1 = df1.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), df1["value"])

        # Construct expected resulting dataframe
        date = []
        for day in range(1, days+1):
            date.append("2019-07-0{} 00:00:00".format(day))

        value = [11.5, 35.5, 59.5, 83.5, 107.5]
        test1 = [(i,j) for i,j in zip(date, value)]
        test = spark.createDataFrame(test1, ["datetime", "value"])
        test = test.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), test["value"])
        test = test.toPandas()

        # Run method
        dataeng = det.DataframeTools(df1)
        result = dataeng.avg_over_period(df1, period="day")
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)

if __name__ == '__main__':
    unittest.main()

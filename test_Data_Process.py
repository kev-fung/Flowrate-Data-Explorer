"""unittest for Data Catalogue

todo:
    Define unittest for thresholding

"""
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
import_mod("Data_Process")
import Data_Process as DP


class TestDataProcess(unittest.TestCase):
    def test_avg_over_period(self):
        """Test if averaging over a day given hourly data works"""

        # Contruct dummy dataframe with datetime
        date, days = [], 5
        for day in range(1, days+1):
            for hour in range(24):
                date.append("2019-07-0{} {}:00:00".format(day, hour))

        value = [i for i in range(days*24)]
        df = [(i, j) for i, j in zip(date, value)]
        df1 = spark.createDataFrame(df, ["datetime", "value"])
        df1 = df1.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), df1["value"])

        # Construct expected resulting dataframe
        date = []
        for day in range(1, days+1):
            date.append("2019-07-0{} 00:00:00".format(day))

        value = [11.5, 35.5, 59.5, 83.5, 107.5]
        test1 = [(i, j) for i, j in zip(date, value)]
        test = spark.createDataFrame(test1, ["datetime", "value"])
        test = test.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), test["value"])
        test = test.toPandas()

        # Run method
        result = DP.avg_over_period(df1, period="day")
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)

    def test_zscore_method(self):
        """Test if zscore method produces 0 mean and 1 stddev in columns"""

        # Construct dummy DataFrame with datetime:
        date, years, months = [], 2, 12
        for year in range(1, years):
            for mon in range(1, months):
                date.append("201{}-0{}-01 00:00:00".format(year, mon))
        value = [i for i in range(years * months)]
        df = [(i, j) for i, j in zip(date, value)]
        df1 = spark.createDataFrame(df, ["datetime", "value"])
        df1 = df1.select(functions.to_timestamp(
            functions.col("datetime").cast("string"), "yyyy-MM-dd HH:mm:ss").alias("datetime"), df1["value"])

        # Run method
        testu = DP.zscore_standard(df1)

        # Test if mean and stddev is 0 and 1
        ll = testu.select(functions.mean("value"), functions.stddev("value")).collect()
        print(ll[0][0], ll[0][1])
        assert np.isclose(0, ll[0][0]) & np.isclose(ll[0][1], 1)


if __name__ == '__main__':
    unittest.main()

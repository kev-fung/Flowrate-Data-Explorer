"""unittest for Data Catalogue

todo:
    Define unittest for removenulls

"""
import unittest
from pandas.util.testing import assert_frame_equal
import os
import sys
import importlib.util
from pathlib import Path
from pyspark.sql import SparkSession
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
import_mod("Data_Catalogue")
import Data_Catalogue as DC


class TestDataCatalogue(unittest.TestCase):
    def test_append_data(self):
        """Test method attaches dataframe to the bottom of the given dataframe"""

        # Construct dummy dataframes
        dummy1 = [("John", 1.0, 2, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, 6)]
        df1 = spark.createDataFrame(dummy1, ["name", "a", "b", "c", "d", "e"])
        dummy2 = [("JJ", 1.0, 2, 3, 8, 5), ("Bizarre", 1.8, 3, 3, 5, 6)]
        df2 = spark.createDataFrame(dummy2, ["name", "a", "b", "c", "d", "e"])

        # Construct expected dataframe
        test1 = [("John", 1.0, 2, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, 6), ("JJ", 1.0, 2, 3, 8, 5),
                 ("Bizarre", 1.8, 3, 3, 5, 6)]
        test = spark.createDataFrame(test1, ["name", "a", "b", "c", "d", "e"])
        test = test.toPandas()

        # Run method
        dataframetools = DC.DataframeTools()
        result = dataframetools.append_data(df1, df2, False)
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)

    def test_null2zero(self):
        """Test if null2zero can convert nulls to zero"""

        # Construct dummy dataframe
        dummy1 = [("John", 1.0, None, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, None)]
        df1 = spark.createDataFrame(dummy1, ["name", "a", "b", "c", "d", "e"])

        # Construct test dataframe
        test1 = [("John", 1.0, 0, 3, 4, 5), ("Snow", 1.3, 3, 4, 5, 0)]
        test = spark.createDataFrame(test1, ["name", "a", "b", "c", "d", "e"])
        test = test.toPandas()

        # Run method
        dataframetools = DC.DataframeTools()
        result = dataframetools.null2zero("b", df1)
        result = dataframetools.null2zero("e", result)
        result = result.toPandas()

        # Test equality
        assert_frame_equal(result, test)


if __name__ == '__main__':
    unittest.main()

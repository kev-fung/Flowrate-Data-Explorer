"""
Kevin Fung
Github: kkf18
"""

import pytest
import pandas
import os
from mlflowrate.classes.base import Base
from pyspark import SparkContext, SQLContext, SparkConf


@pytest.fixture(scope="session")
def spark_context():
    conf = (SparkConf().setMaster("local[2]").setAppName("mf-flow-rate-test"))
    spark_context = SparkContext(conf=conf)
    return spark_context


@pytest.fixture(scope="session")
def sql_context(spark_context):
    sql_context = SQLContext(spark_context)
    return sql_context


@pytest.fixture(scope="session")
def data_frame_names():
    names = os.listdir("resource")
    return [filename.split(".")[0] for filename in names]


@pytest.fixture(scope="session")
def data_frames(spark_context, sql_context, data_frame_names):
    data_frames = [(name, pandas.read_csv("resources/{}.csv".format(name))) for name in data_frame_names]
    spark_data_frames = {name: create_spark_data_frame(sql_context, data_frame) for (name, data_frame) in data_frames}
    return spark_data_frames


@pytest.fixture(scope="session")
def base_data(data_frames):
    base_data = BaseData(dfs=data_frames)
    return base_data


def create_spark_data_frame(sql_context, data_frame):
    return sql_context.createDataFrame(data_frame)

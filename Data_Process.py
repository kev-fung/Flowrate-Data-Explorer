"""Subclass of Data for Processing Data

This module contains methods which would preprocess/augment
spark dataframes in the Azure Databricks environment.

@author: Kevin Fung
"""
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from googletrans import Translator
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


def translate_col(head, df, src='no', dest='en'):
    """Translate a string column of a Spark DataFrame from src language to dest language
    Args:
        head (str): Column name
        df (Spark DataFrame): Spark DataFrame given
        src (str): Source language to translate from
        dest (str): Target language to translate to

    Returns:
        Spark DataFrame: Spark DataFrame with translated column
        Dictionary: dictionary of distinct strings translated
    """
    # Translate descriptions
    translator = Translator()
    # Select distinct comments
    n = df.select(df[head]).distinct().rdd.map(lambda x: x[head]).collect()
    # Make a dictionary to translate distinct comments
    translation_dict = {col: translator.translate(col, src=src, dest=dest).text for col in n}
    # Utilise spark method and replace all norweigan comments with translated ones
    new_df = df.na.replace(translation_dict, 1, head)

    return new_df, translation_dict


def ts_overlay_records(df_ts, df_records, head, filt_dict=None, translate_dict=None):
    """Given a ts dataframe, make a new column and match corresponding records according to time.
       Pass in filt_dict and translate_dict to be able to group up similar records.
    Args:
      df_ts (Spark DataFrame): Time series DataFrame
      df_records (Spark DataFrame): Time series DataFrame whose values contain records (e.g. string descriptions)
      head (str): Column name of the records in df_records
      filt_dict (dict): Dictionary of similar substrings found in similar records and it's new collective record
      translate_dict (dict): Dictionary of translated distinct records

    Returns:
      Spark DataFrame: Spark DataFrame with new column of time matched records
    """
    # Make a new column where datetime precision: daily. (because interferences are recorded daily)
    df = df_ts.select(
        df_ts["datetime"].alias("datetime_orig"),
        (F.round(F.unix_timestamp(F.col("datetime")) / 86400) * 86400).cast("timestamp").alias("datetime"),
        df_ts["value"]
    )

    new_df = df.join(df_records, on=['datetime'], how='left_outer')

    if (filt_dict is not None) and (translate_dict is not None):
        print("\nCollecting similar comments given the filt_dict...")
        group_dict = {}
        for comment in translate_dict.values():
            for abrv, group_comment in filt_dict.items():
                if abrv in comment:
                    group_dict[comment] = group_comment
                    # print(comment, " : ", group_comment)

        new_df = new_df.withColumn("Grouped", new_df[head])
        new_df = new_df.na.replace(group_dict, 1, "Grouped")
        new_df = new_df.na.fill("No Records", "Grouped")

    new_df = new_df.na.fill("No Records", head)
    new_df = new_df.drop("datetime")
    new_df = new_df.withColumnRenamed("datetime_orig", "datetime")

    print("\nOverlaid records onto timeseries: You may need to remove/merge duplicates!")
    print("\nDuplicates found: ", new_df.dropDuplicates(["datetime"]).count())

    return new_df


def numerate_desc(df, head):
    """Make new column for a Spark DataFrame which numerically discretises a descriptive column.
    Args:
        df (Spark DataFrame): Spark Dataframe, FORMAT: datetime|value|Description|Grouped
        head (str): Name of the column to discretise

    Returns:
        new_df (Spark DataFrame): DataFrame with column of discretised values of the descriptive column
    """
    # Make column by numerically discretising distinct comments
    grouped_desc_list = df.select(df[head]).distinct().rdd.map(lambda x: x[head]).collect()

    # Convert comments to discrete values
    numerate = {str(val): str(i) for i, val in enumerate(grouped_desc_list)}

    new_df = df.withColumn("Discrete_str", df[head])
    new_df = new_df.na.replace(numerate, 1, "Discrete_str")
    new_df = new_df.withColumn("Discrete", new_df["Discrete_str"].cast(IntegerType())).drop("Discrete_str")
    new_df = new_df.na.fill(0, "Discrete")

    return new_df


def merge_duplicate(df, sqlContext):
    """Collect up duplicated datetimes with different descriptions, and merge the descriptions together.
      Args:
        df (Spark DataFrame): Spark DataFrame, FORMAT: datetime|value|description|groupedDescription
        sqlContext (Spark Obj): required for Spark RDD creation

      Returns:
        Spark DataFrame: Spark DataFrame with merged descriptions
    """
    reduced = df \
        .rdd \
        .map(lambda row: (row[0], [(row[1], row[2], row[3])])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda row: (
            row[0],  # key i.e. datetime
            row[1][0][0],  # sum(row[1][0]) / len(row[1][0]),       #value, take the average of the values
            ','.join([str(e[1]) for e in row[1]]),  # join up the descriptions
            ','.join([str(e[2]) for e in row[1]])  # join up the grouped descriptions
            )
         )
    schema_red = df.schema
    new_df = sqlContext.createDataFrame(reduced, schema_red).orderBy("datetime")
    if new_df.count() > new_df.dropDuplicates(["datetime"]).count():
        raise ValueError('\nData still has duplicates!')
    print("\nDuplicates have been merged")

    return new_df


def avg_over_period(df, period="day"):
    """Given a dataframe with datetime column, average over days, weeks, months or years and return new dataframe.
        Args:
            df (Spark DataFrame): input Spark Dataframe, FORMAT: datetime|value|etc.
            period (str): period for averaging over, EXAMPLES: day, week, month, year

        Returns:
            Spark DataFrame: Spark DataFrame whose values are averaged according to period
    """
    new_df = df.withColumn(period, F.date_trunc(period, df.datetime))
    new_df = new_df \
        .groupBy(period) \
        .agg(F.avg("value")) \
        .orderBy(period)

    new_df = new_df.withColumnRenamed("avg(value)", "value")
    new_df = new_df.withColumnRenamed(period, "datetime")

    return new_df


def threshold_avg(df, prange=0.7):
    """Remove anomalous values based on a defined percentage range from the weekly averaged mean of data
      Args:
          df (Spark DataFrame): input Spark DataFrame, FORMAT: datetime|value|etc.
          prange (double): percentage range (of one direction) away from mean

      Returns:
          Spark DataFrame: Spark DataFrame with thresholded samples
    """
    weekly_avg_dfs = avg_over_period(df, "week")
    mean, std = weekly_avg_dfs.select(F.mean("value"), F.stddev("value")).first()

    new_df = df.where((df.value >= (1-prange)*mean) & (df.value <= (1+prange)*mean)).orderBy("datetime")

    remove_count = df.select("value").count() - new_df.select("value").count()
    print("\nThresholding has removed: ", remove_count, " samples from dataframe")

    return new_df


def zscore_standard(df, exceptions=None):
    """Apply z-score standardisation to the features in Spark DataFrame
    Args:
        df (Spark DataFrame): input Spark DataFrame, FORMAT: datetime|feature1|feature2...
        exceptions (list(str)): Explicitly state which features to not normalise
  
    Returns:
        Spark DataFrame: Spark DataFrame with normalised features
    """
    if exceptions is None:
        exceptions = ["datetime"]

    orig_heads = df.schema.names  # schema names gets updated dynamically! So must be separate from the loop

    for head in orig_heads:
        if head in exceptions: continue
        mean, std = df.select(F.mean(head), F.stddev(head)).first()
        df = df.withColumn("{}_".format(head), (F.col(head) - mean) / std)
        df = df.drop(head)
        df = df.withColumnRenamed("{}_".format(head), head)

    return df


def threshold_minmax(df, head, upper, lower):
    """ Remove any rows in Spark DataFrame whose values in head is out of provided bounds
    Args:
        df (Spark DataFrame): input Spark DataFrame
        head (str): name of column
        upper (double): upper bound limit
        lower (double): lower bound limit

    Returns:
        Spark DataFrame: Spark DataFrame with removed rows whose values in head is out of bounds
    """
    assert upper > lower, "upper bounds smaller than lower bounds!"
    return df.where((df[head] >= lower) & (df[head] <= upper))


def threshold_minmax_round(df, head, upper, lower):
    """ Round any values in Spark DataFrame out of provided bounds to the boundary limits
    Args:
        df (Spark DataFrame): input Spark DataFrame
        head (str): name of column
        upper (double): upper bound limit
        lower (double): lower bound limit

    Returns:
        Spark DataFrame: Spark DataFrame with out of boundary values rounded to its nearest limit
    """
    assert upper > lower, "upper bounds smaller than lower bounds!"
    df1 = df.withColumn("clipped", F.when(df[head] > upper, upper).when(df[head] < lower, lower).otherwise(df[head]))
    df1 = df1.drop(head).withColumnRenamed("clipped", head)
    return df1

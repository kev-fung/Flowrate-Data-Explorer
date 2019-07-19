# Import Modules
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType, BooleanType, ArrayType, IntegerType, TimestampType, DoubleType
from pyspark.sql.functions import coalesce, lit, col, lead, lag
from pyspark.sql.functions import stddev, mean
from pyspark.sql import SQLContext
from pyspark.sql.window import Window

from operator import add
from functools import reduce

from googletrans import Translator

# Standard Python Modules
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import re


class DataframeTools:
    """Parent class for manipulating spark dataframes. """

    def __init__(self, df):
        self.df = df

    def update_df(self, new_df):
        """Replace class dataframe with a new one.

        Args:
          new_df (dataframe): New dataframe

        """

        self.df = new_df

    def append_data(self, new_df):
        """Append another dataframe below the current one.
        New dataframe must have the same columns as the original dataframe.

        Args:
          new_df (dataframe): New dataframe

        """

        assert self.df.schema == new_df.schema, "Column headers must match!"

        print("Current samples: ", self.df.count())
        print("Appending samples: ", new_df.count())

        self.df = self.df.union(new_df)
        print("Joined samples: ", self.df.count())
        print("")

    def is_null(self, dfs):
        """Check all columns in dataframe for null values.

        Args:
          dfs (dataframe): Dataframe to be checked on

        """

        print("Number of samples with a null value across columns:")
        for col in dfs.schema:
            head = col.name
            # print(dfs[head].isNull() == True)
            print(head, dfs.where(dfs[head].isNull() == True).count())
        print("")

    def null2zero(self, head, dfs):
        """Change null values to zero in column

        Args:
          head (str): Column name
          dfs (dataframe): Dataframe given

        Returns:
          dataframe with replaced null values

        """

        print("Replacing null values with zero...\n")
        dfs = dfs.na.fill(0, head)
        return dfs

    def translate_col(self, head, dfs, src='no', dest='en'):
        """Translate a string column of a dataframe from src language to dest language

        Args:
          head (str): Column name
          dfs (dataframe): Dataframe given
          src (str): Source language to translate from
          dest (str): Target language to translate to

        Returns:
          new_dfs (dataframe): dataframe with translated column
          translation_dict (dict): dictionary of distinct strings translated

        """

        # Translate descriptions
        translator = Translator()
        # Select distinct comments
        n = dfs.select(dfs[head]).distinct().rdd.map(lambda x: x[head]).collect()
        # Make a dictionary to translate distinct comments
        translation_dict = {col: translator.translate(col, src=src, dest=dest).text for col in n}
        # Utilise spark method and replace all norweigan comments with translated ones
        new_dfs = dfs.na.replace(translation_dict, 1, head)
        return new_dfs, translation_dict

    def ts_overlay_records(self, df_ts, df_records, head, filt_dict=None, translate_dict=None):
        """Given a ts dataframe, make a new column and match corresponding records according to time.
           Pass in filt_dict and translate_dict to be able to group up similar records.

        Args:
          df_ts (dataframe): Time series dataframe
          df_records (dataframe): Time series dataframe whose values contain records (e.g. string descriptions)
          head (str): Column name of the records in df_records
          filt_dict (dict): Dictionary of similar substrings found in similar records and it's new collective record
          translate_dict (dict): Dictionary of translated distinct records

        Returns:
          new_df (dataframe): dataframe containing new column of time correlated records to the time series and other useful columns

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
        print("Duplicates found: ", new_df.dropDuplicates(["datetime"]).count())

        return new_df

    def discretise_col(self, dfs, head):
        """Make new column for a dataframe which numerically discretises a descriptive column.

        Args:
          dfs (dataframe): Dataframe which contains: datetime, value, Description, Grouped
          head (str): Name of the column to discretise

        Returns:
          dataframe with column of discretised values of the descriptive column

        """

        # Make column by numerically discretising distinct comments
        grouped_desc_list = dfs.select(dfs[head]).distinct().rdd.map(lambda x: x[head]).collect()

        # Convert comments to discrete values
        numerate = {str(val): str(i) for i, val in enumerate(grouped_desc_list)}

        new_dfs = dfs.withColumn("Discrete_str", dfs[head])

        new_dfs = new_dfs.na.replace(numerate, 1, "Discrete_str")
        new_dfs = new_dfs.withColumn("Discrete", new_dfs["Discrete_str"].cast(IntegerType())).drop("Discrete_str")
        new_dfs = new_dfs.na.fill(0, "Discrete")

        return new_dfs

    def add_year_col(self, dfs):
        """Add column of year of date to dataframe.

        Args:
          dfs (dataframe): dataframe with datetime

        Returns:
          dataframe with column for year.

        """

        return dfs.withColumn("year", F.year(F.col("datetime")))

    def add_quart_col(self, dfs):
        """Add column of the quarterly period to dataframe.

        Args:
          dfs (dataframe): dataframe with datetime

        Returns:
          dataframe with column for quarterly period.

        """

        return dfs.withColumn("quarter", F.quarter(F.col("datetime")))

    def merge_duplicate(self, dfs, sqlContext):
        """Collect up duplicated datetimes with different descriptions, and merge the descriptions together.

          Args:
            dfs (dataframe): dataframe in the format: datetime|value|description|groupedDescription
            sqlContext (spark object): required for spark RDD creation

          Returns:
            dataframe with merged descriptions

        """

        reduced = dfs \
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

        schema_red = dfs.schema

        new_dfs = sqlContext.createDataFrame(reduced, schema_red).orderBy("datetime")

        if new_dfs.count() > new_dfs.dropDuplicates(["datetime"]).count():
            raise ValueError('Data has duplicates')

        return new_dfs

    def avg_over_period(self, dfs, period="day"):
        """Given a dataframe with datetime column, average over days, weeks, months or years and return new dataframe.

          Args:
            dfs (dataframe): dataframe in the format of: datetime|value|...|...
            period (str): period for averaging over, e.g. day, week, month, year...

          Returns:
            dataframe in format of: datetime|value  , where value is the averaged value over the period

        """

        dfs_new = dfs.withColumn(period, F.date_trunc(period, dfs.datetime))

        dfs_new = dfs_new \
            .groupBy(period) \
            .agg(F.avg("value")) \
            .orderBy(period)

        dfs_new = dfs_new.withColumnRenamed("avg(value)", "value")
        dfs_new = dfs_new.withColumnRenamed(period, "datetime")

        return dfs_new

    def threshold(self, dfs, thresh=0.7):
        """Remove any anomalous values based on a threshold cutoff from the mean of the dataset (which has been weekly averaged)

          Args:
            dfs (dataframe): input dataframe in format of: datetime|value|...
            thresh (double): the percentage range from the mean for thresholding

          Returns:
            dataframe with thresholded samples

        """

        weekly_avg_dfs = self.avg_over_period(dfs, "week")
        mean, std = weekly_avg_dfs.select(F.mean("value"), F.stddev("value")).first()

        new_dfs = dfs.where((dfs.value >= (1 - thresh) * mean) & (dfs.value <= (1 + thresh) * mean)).orderBy("datetime")

        remove_count = dfs.select("value").count() - new_dfs.select("value").count()
        print("\nThresholding has removed: ", remove_count, " samples from dataframe")

        return new_dfs


class GroupDataTools(DataframeTools):
    """Subclass of Dataframe tools to reorganise dataframes into dictionaries.
      Tools for visualisation and preprocessing included.
    """

    def __init__(self, df, df_dict={}):
        super().__init__(df)
        self.df_dict = df_dict
        self.headers = [h.name for h in df.schema]

    def groupdata(self, dict_group_head, x_head, y_head):
        """Collect rows which contain the same value in dict_group_head column and
        put them into a dictionary.

        Keys: distinct value, Value: dataframe whose rows contain distinct value

        Args:
          dict_group_head (str): column to split the dataframe
          x_head (str): first column (normally time) to reconstruct corresponding dataframe
          y_head (str): second column (normally value) to reconstruct corresponding dataframe

        """

        # Split dataframe up by given header
        assert dict_group_head in self.headers, "Header does not exist in dataframe!"

        unq_items = self.df.select(dict_group_head).distinct()
        n = unq_items.rdd.map(lambda x: x[dict_group_head]).collect()

        for key in n:
            self.df_dict[key] = self.df.orderBy(x_head).where(self.df[dict_group_head] == key).select(self.df[x_head],
                                                                                                      self.df[y_head])

    def splitdata_dict(self, regex):
        """Make a dictionary from df_dict given a conditional substring which matches within the keys of df_dict.
        E.g. Return a dictionary of a certain oilwell given from a code in the tag name.

        Args:
          regex (str): substring to match with keys, written in wildcard format for regular expressions

        """

        out_dict = {}

        for (key, val) in self.df_dict.items():
            for reg in regex:
                if re.match(reg, key):
                    out_dict[key] = val

        return out_dict

    def decode_keys(self, in_dict, decode_dict):
        """Replace the old keys with new key definitions.

          Args:
            in_dict (dict): input oilwell dictionary containing dataframes
            decode_dict (dict): dictionary of old keys with new definitions

          Returns: None

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
        print("")

        return new_dict

    def plot_ts(self, title, x_head, y_head, ts_df_list, label_list=["value"], **kwargs):
        """Plot multiple timeseries dataframe onto a figure, x axis = time, y axis = value.

        Args:
          title (str): Name of dataframe
          x_head (str): Name of column to be plotted along x axis
          y_head (str): Name of column to be plotted along y axis
          ts_df_list (list): list of timeseries dataframes to plot
          label_list (list): list of plot labels

          **kwargs: Additional options:
            overlay (str): header of column containing descriptions
            overlay_dfs (dataframe): dataframe containing the descriptions,  dataframe format: datetime|value|overlay|...
            plot_yearly (list): plot yearly data on separate axes by passing in list of years
            plot_quarterly (list): plot quarterly data of given list of years

        """

        if "plot_yearly" not in kwargs.keys():
            fig, ax = plt.subplots(1, 1, figsize=(24, 8))

            for ts_df, lab in zip(ts_df_list, label_list):
                ts_pd = ts_df.orderBy(x_head).toPandas()

                y = ts_pd[y_head].tolist()
                x = ts_pd[x_head].tolist()

                ax.plot(x, y, "-", label=lab)
                ax.grid(True)

            if ("overlay" in kwargs.keys()) and ("overlay_dfs" in kwargs.keys()):
                # the overlay dataframe should have a value column which is the same as another column being plotted!
                self.__overlay_plot(ax, kwargs["overlay"], kwargs["overlay_dfs"])

            ax.set_title(title, fontsize=16)
            ax.set_xlabel(x_head, fontsize=16)
            ax.set_ylabel(y_head, fontsize=16)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m-%d"))

            ax.legend(loc="best")

            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        else:
            ts_df_years_list = [self.add_year_col(df) for df in ts_df_list]

            # double list: dfs_years[df][year]
            # check if input years list is valid in the dfs column of years
            dfs_years = [[df.where(df.year == y).toPandas() for y in kwargs["plot_yearly"]] for df in ts_df_years_list]

            plots = len(kwargs["plot_yearly"])

            fig, axs = plt.subplots(plots, 1, figsize=(24, 8 * plots))
            axs.flatten()

            for df, lab in zip(dfs_years, label_list):
                for year_plot, ax, year in zip(df, axs, kwargs["plot_yearly"]):
                    ts_pd = year_plot.sort_values(x_head)

                    y = ts_pd[y_head].tolist()
                    x = ts_pd[x_head].tolist()

                    ax.plot(x, y, "-", label=lab)
                    ax.grid(True)
                    ax.set_title("{}, {}".format(title, year), fontsize=16)
                    ax.legend(loc="best")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
                    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m-%d"))

                    ax.set_xlabel(x_head, fontsize=16)
                    ax.set_ylabel(y_head, fontsize=16)

            if ("overlay" in kwargs.keys()) and ("overlay_dfs" in kwargs.keys()):
                overlay_dfs_ = self.add_year_col(kwargs["overlay_dfs"])
                overlay_dfs_years = [overlay_dfs_.where(overlay_dfs_.year == y) for y in kwargs["plot_yearly"]]

                for year_plot, ax in zip(overlay_dfs_years, axs):
                    self.__overlay_plot(ax, kwargs["overlay"], year_plot)

            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        if "plot_quarterly" in kwargs.keys():
            ts_df_y_list = [self.add_year_col(df) for df in ts_df_list]
            ts_df_yq_list = [self.add_quart_col(df) for df in ts_df_y_list]

            # triple list: dfs_years[df][year][quarter]
            # check if input years list is valid in the dfs column of years
            dfs_yq = []
            for df in ts_df_yq_list:
                dfs_y = []
                for y in kwargs["plot_quarterly"]:
                    dfs_q = []
                    for q in range(1, 5):
                        dfs_q.append(df.where((df.year == y) & (df.quarter == q)).toPandas())
                    dfs_y.append(dfs_q)
                dfs_yq.append(dfs_y)

            plots = len(kwargs["plot_quarterly"])*4

            fig, axs = plt.subplots(plots, 1, figsize=(24, 8*plots))
            axs.flatten()

            for dfs_y, lab in zip(dfs_yq, label_list):
                for dfs_q in dfs_y:
                    for q_plot, ax, year, quarter in zip(dfs_q, axs, kwargs["plot_quarterly"], range(1, 5)):
                        ts_pd = q_plot.sort_values(x_head)

                        y = ts_pd[y_head].tolist()
                        x = ts_pd[x_head].tolist()

                        ax.plot(x, y, "-", label=lab)
                        ax.grid(True)
                        ax.set_title("{}, {}, Q{}".format(title, year, quarter), fontsize=16)
                        ax.legend(loc="best")
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
                        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m-%d"))

                        ax.set_xlabel(x_head, fontsize=16)
                        ax.set_ylabel(y_head, fontsize=16)

            if ("overlay" in kwargs.keys()) and ("overlay_dfs" in kwargs.keys()):
                overlay_dfs_ = self.add_year_col(kwargs["overlay_dfs"])
                overlay_dfs_ = self.add_quart_col(overlay_dfs_)

                overlay_dfs_yq = [[overlay_dfs_.where((overlay_dfs_.year == y) & (overlay_dfs_.quarter == q)) for q in
                                   range(1, 5)] for y in kwargs["plot_quarterly"]]

                for year in overlay_dfs_yq:
                    for q, ax in zip(year, axs):
                        self.__overlay_plot(ax, kwargs["overlay"], q)

            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        return fig

    def __overlay_plot(self, ax, overlay, overlay_dfs):
        """Private function to plot descriptive data over linear graphs

        Args:
            ax (axes): matplotlib axes to plot on
            overlay (str): header name of overlay column in dataframe
            overlay_dfs (dataframe): dataframe containing the overlay column

        """

        dfs_pd = overlay_dfs.toPandas()
        groups = dfs_pd.groupby(overlay)

        for name, group in groups:
            if name == "No Records": continue
            ax.plot(group.datetime, group.value, marker='o', linestyle='', label=name)
            ax.legend(loc="best")

    def weighted_average(self, ts_df, x_head, y_head, offsets, weights):
        """Produce rolling average results of the given ts data with the given specs.

          Args:
            ts_df (dataframe): timeseries dataframe
            x_head (str): header name of x axis in timeseries (e.g. datetime)
            y_head (str): header name of y axis in timeseries (e.g. value)
            offsets (list): list of adjacent values to consider
            weights (list): list of weights applied to offsets

        """
        window = Window.orderBy(x_head)
        v = col(y_head)

        assert len(weights) == len(offsets)

        def value(i):
            if i < 0: return lag(v, -i).over(window)
            if i > 0: return lead(v, i).over(window)
            return v

        values = [coalesce(value(i) * w, lit(0)) / len(offsets) for i, w in zip(offsets, weights)]

        return reduce(add, values, lit(0))

    def view_moving_avg(self, title, x_head, y_head, y_label, ts_df, offsets, weights):
        """Wrapper function to view the moving average of the given ts data

          Args:
            title (str): Title of graph
            x_head (str): Header name for x column data
            y_head (str): Header name for y column data
            y_label (str): y axis label
            ts_df (dataframe): Time series dataframe
            offsets (list): list of adjacent values to consider
            weights (list): list of weights applied to offsets

        """

        avg = ts_df.withColumn("avg", self.weighted_average(ts_df, offsets, weights)).drop(y_head)
        avg = avg.select(avg[x_head],
                         avg["avg"].alias(y_head))

        self.plot_ts(title, y_label, [avg])

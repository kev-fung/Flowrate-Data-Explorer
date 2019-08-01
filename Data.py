"""Parent class of Data_ subclasses

This module contains methods which would be applicable during any of the
pipeline stages (Catalogue, Process, Analytics) in the Azure Databricks
environment.

@author: Kevin Fung
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyspark.sql import functions as F


class Data:
    """Parent class containing general data methods"""
    def __init__(self, df):
        self.df = df

    def plot_ts(self, title, x_head, y_head, ts_df_list, label_list=None, **kwargs):
        """Plot multiple timeseries dataframe onto a figure, x axis = time, y axis = value.

        Args:
            title (str): Name of dataframe
            x_head (str): Name of column to be plotted along x axis
            y_head (str): Name of column to be plotted along y axis
            ts_df_list (list): list of timeseries dataframes to plot
            label_list (list): list of plot labels

            **kwargs:
                overlay (str): header of column containing descriptions
                overlay_dfs (dataframe): dataframe containing the descriptions, FORMAT: datetime|value|overlay|etc.
                plot_yearly (list): plot yearly data on separate axes by passing in list of years
                plot_quarterly (list): plot quarterly data of given list of years

        """
        if label_list is None:
            label_list = ["value"]

        if "plot_yearly" not in kwargs.keys():
            fig, ax = plt.subplots(1, 1, figsize=(24, 8))

            for ts_df, lab in zip(ts_df_list, label_list):
                ts_pd = ts_df.orderBy(x_head).toPandas()

                y = ts_pd[y_head].tolist()
                x = ts_pd[x_head].tolist()

                ax.plot(x, y, ".--", label=lab)
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

                    ax.plot(x, y, ".--", label=lab)
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

            for dfs_y, lab in zip(range(len(dfs_yq)), label_list):
                for dfs_q, year in zip(range(len(dfs_yq[0])), kwargs["plot_quarterly"]):
                    for quarter in range(1, 5):
                        ts_pd = dfs_yq[dfs_y][dfs_q][quarter - 1].sort_values(x_head)

                        y = ts_pd[y_head].tolist()
                        x = ts_pd[x_head].tolist()

                        axs[(quarter - 1) + (4 * dfs_q)].plot(x, y, ".--", label=lab)
                        axs[(quarter - 1) + (4 * dfs_q)].grid(True)
                        axs[(quarter - 1) + (4 * dfs_q)].set_title("{}, {}, Q{}".format(title, year, quarter),
                                                                   fontsize=16)
                        axs[(quarter - 1) + (4 * dfs_q)].legend(loc="best")

                        axs[(quarter - 1) + (4 * dfs_q)].set_xlabel(x_head, fontsize=16)
                        axs[(quarter - 1) + (4 * dfs_q)].set_ylabel(y_head, fontsize=16)

            if ("overlay" in kwargs.keys()) and ("overlay_dfs" in kwargs.keys()):
                overlay_dfs_ = self.add_year_col(kwargs["overlay_dfs"])
                overlay_dfs_ = self.add_quart_col(overlay_dfs_)

                overlay_dfs_yq = [[overlay_dfs_.where((overlay_dfs_.year == y) & (overlay_dfs_.quarter == q)) for q in
                                   range(1, 5)] for y in kwargs["plot_quarterly"]]

                for year in range(len(overlay_dfs_yq)):
                    for quarter in range(1, 5):
                        self.__overlay_plot(axs[(quarter - 1) + (4 * year)], kwargs["overlay"],
                                            overlay_dfs_yq[year][quarter - 1])

            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

        return fig

    def __overlay_plot(self, ax, overlay, overlay_df):
        """Private function to plot descriptive data over linear graphs

        Args:
            ax (axes): matplotlib axes to plot on
            overlay (str): header name of overlay column in dataframe
            overlay_df (dataframe): dataframe containing the overlay column

        Returns:
            None

        """
        df_pd = overlay_df.toPandas()
        groups = df_pd.groupby(overlay)

        for name, group in groups:
            if name == "No Records":
                continue
            ax.plot(group.datetime, group.value, marker='o', linestyle='', label=name)
            ax.legend(loc="best")

    def add_year_col(self, df):
        """Add column of year of date to dataframe.

        Args:
          df (dataframe): dataframe with datetime

        Returns:
          dataframe with column for year.

        """
        return df.withColumn("year", F.year(F.col("datetime")))

    def add_quart_col(self, df):
        """Add column of the quarterly period to dataframe.

        Args:
          df (dataframe): dataframe with datetime

        Returns:
          dataframe with column for quarterly period.

        """
        return df.withColumn("quarter", F.quarter(F.col("datetime")))

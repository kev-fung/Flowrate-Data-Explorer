"""Subclass of Data for Modelling Data

This module contains methods for machine learning applications
in the Azure Databricks environment.

@author: Kevin Fung
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
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


def correlate(df):
    """Plot Pearsons Correlation Matrix given a Spark DataFrame
    Args:
        df (Spark DataFrame): input Spark DataFrame

    Returns:
        fig (Figure Obj): Figure object containing plots
    """
    df_pd = df.toPandas()
    fig, ax = plt.subplots(figsize=(14,12))
    colormap = plt.cm.RdBu
    ax.set_title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df_pd.corr(), linewidths=0.1, vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True, ax=ax)

    return fig


class ProcessDatasets:
    """Helper class to split and clean spark dataframes elegantly"""

    def __init__(self, df=None, log=None):
        self.df = df
        self.orig_df = df
        if log is None:  # may want to keep track of transformations
            self.log = []

    def set_df(self, df):
        self.df = df
        self.orig_df = df

    def get_df(self, pandas=False):
        if pandas:
            return self.df.toPandas()
        else:
            return self.df

    def get_date(self, pandas=False):
        assert "datetime" in self.df.schema.names, "class member df does not have a datetime column!"
        if pandas:
            return self.df.select("datetime").toPandas()
        else:
            return self.df.select("datetime")

    def apply_transformations(self, trans, reset_df_to_orig=True, args=None):
        if args is None:
            args = []

        if reset_df_to_orig is True:
            self.df = self.orig_df
            print("\nReset dataframe to original")

        for transform in trans:
            self.df = transform(self.df, *args)
        print("Dataframe has been transformed")

    def split_df(self, features, label="label", pandas=False):
        X = self.df.select(*features)
        y = self.df.select(label)

        if pandas:
            X = X.toPandas()
            y = y.toPandas()

        print("Dataframe has been split and returned")
        return X, y


def aic(y, y_pred, k):
    resid = y - y_pred
    sse = sum(resid ** 2)
    return 2 * k - 2 * np.log(sse)


def kfold_scores(model, X, y, k=10, return_info=False, v=False):
    """ X: entire dataset of features
        y: entire dataset of labels
    """
    X, y = X.values, y.values
    kf_feat_datasets, kf_lab_datasets = [], []
    kinfo = {"models": [],
             "aic": [],
             "rms": [],
             "eval_metrics": []
             }
    sum_aic, sum_rms = 0, 0
    m = len(y)

    kf = KFold(n_splits=k)

    for train_id, val_id in kf.split(X, y):
        kf_feat_datasets.append([X[train_id], X[val_id]])
        kf_lab_datasets.append([y[train_id], y[val_id]])

    for n in range(kf.get_n_splits()):
        kmodel = clone(model)
        kmodel.fit(kf_feat_datasets[n][0], kf_lab_datasets[n][0])
        kinfo["models"].append(kmodel)
        y_val = kmodel.predict(kf_feat_datasets[n][1])
        if v:
            print("\nFitted Parameters: ", kmodel.coef_)

        # AIC Estimator:
        aic_ = aic(kf_lab_datasets[n][1], y_val, 5)
        sum_aic += aic_
        kinfo["aic"].append(aic_)
        if v:
            print("AIC Estimator: ", aic_)

        # RMS Estimator:
        rms_ = ((kf_lab_datasets[n][1] - y_val) * 2).mean() * .5
        sum_rms += rms_
        kinfo["rms"].append(rms_)
        if v:
            print("RMS Estimator: ", rms_)

    avg_aic = sum_aic / m
    avg_rms = sum_rms / m
    print("\nModel Evaluation")
    print("-----------------")
    print("Number of folds: ", k)
    print("Averaged AIC: ", avg_aic)
    print("Averaged RMS: ", avg_rms)

    kinfo["eval_metrics"].append(avg_aic)
    kinfo["eval_metrics"].append(avg_rms)

    if return_info is True:
        return kinfo

#
# def hype_tuning(model, df, features, hype_range, trans, kfolds, graph=True):
#     rms = []
#     dataset = ProcessDatasets(df)
#
#     for param in hype_range:
#         dataset.apply_transformations(trans)
#         X, y = dataset.split_df(features)
#         X, y = X, np.squeeze(y)
#
#         kinfo = kfold_scores(model, X, y, k=kfolds, return_info=True)
#         rms.append(kinfo["eval_metrics"][1])
#
#     if graph:
#         fig, ax = plt.subplots(1, 1, figsize=(24, 8))
#         ax.plot(upper, rms, "x-", label="hyp tune")
#         ax.set_xlabel("Parameter Range", fontsize=16)
#         ax.set_ylabel("RMS Error", fontsize=16)
#         ax.set_title("Varying Parameter against RMS Error of model", fontsize=16)
#         ax.legend(loc="best")
#         ax.grid(True)
#         display(fig)
#
#     return rms

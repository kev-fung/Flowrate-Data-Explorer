"""
Kevin Fung
Github: kkf18
"""
from mlflowrate.classes.base import BaseData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression


class DataExplore(BaseData):
    def __init__(self, datasets):
        super().__init__(datasets=datasets)
        self.results = {}  # dictionary of results objects

    def status(self):
        print("\nDatasets Available: ")
        print("~~~~~~~~~~~~~~~~~~~")
        for dname, dobj in self.datasets.items():
            if dobj.is_pandas:
                print("{0}  |  Samples {1}  |  Num Features: {2}  |  Dates: {3}  to  {4}".format(dname,
                                                                                                 dobj.X.shape[0],
                                                                                                 dobj.X.shape[1],
                                                                                                 dobj.date[
                                                                                                     "datetime"].head(
                                                                                                     1).dt.strftime(
                                                                                                     '%Y/%m/%d').values[
                                                                                                     0],
                                                                                                 dobj.date[
                                                                                                     "datetime"].tail(
                                                                                                     1).dt.strftime(
                                                                                                     '%Y/%m/%d').values[
                                                                                                     0]))
            else:
                print("{0}  |  Samples {1}  |  Num Features: {2} ".format(dname,
                                                                          dobj.y.count(),
                                                                          len(dobj.X.columns())))
        print("\nResults Collected: ")
        print("~~~~~~~~~~~~~~~~~")
        for res, resobj in self.results.items():
            print("{0}  |  R2 {1}  |  RMSE {2} ".format(res, resobj.r2, resobj.rmse))

    def benchmark(self, date, ytrue, ypred):
        self.results["benchmark"] = Result(date=date, ytrue=ytrue, ypred=ypred)

    def input_results(self, result_name, date=None, ypred=None, ytrue=None, r2=0.0, rmse=0.0):
        #       wrapper function
        self.results[result_name] = Result(date, ypred, ytrue, r2, rmse)

    def validate(self, model, traindataset, param_grid, outer=10, inner=5, verb=False):

        X = traindataset.X
        y = np.squeeze(traindataset.y)

        outer_cv = KFold(n_splits=outer, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=inner, shuffle=True, random_state=42)
        scoring = {'r2': 'r2',
                   'rmse': make_scorer(mean_squared_error)}

        # Pass the gridSearch estimator to cross_val_score
        clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring=scoring, refit='r2')
        scores = cross_validate(clf, X=X, y=y, scoring=scoring, cv=outer_cv)
        avgr2 = scores['test_r2'].mean()
        avgrmse = np.sqrt(scores['test_rmse']).mean()

        if verb:
            print("R-Squared: ", avgr2)
            print("RMSE: ", avgrmse)

        return avgr2, avgrmse

    def evaluate(self, model, traindataset, testdataset, param_grid, splits=10, testunc=0.15):

        traindate = traindataset.date
        trainX = traindataset.X
        trainy = np.squeeze(traindataset.y)

        testdate = testdataset.date
        testX = testdataset.X
        testy = np.squeeze(testdataset.y)

        kcv = KFold(n_splits=splits, shuffle=True, random_state=42)
        clf = GridSearchCV(model, param_grid, cv=kcv)
        trainpred = clf.fit(trainX, trainy).predict(trainX)
        model_params = clf.best_params_
        testpred = clf.predict(testX)

        trainpred = np.squeeze(trainpred)
        testpred = np.squeeze(testpred)

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        self.trainplot(traindate, trainpred, trainy, axs=axs[0])
        self.evalplot(traindate, testdate, trainy, testy, testpred, testunc=testunc, axs=axs[1])
        display(fig)
        return model_params

    def trainplot(self, traindate, trainpred, trainy, axs=None):
        trainsse = ((trainpred - trainy) ** 2).sum()
        trainrmse = np.sqrt(sse / trainy.shape[0])

        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(12, 4))

        axs.plot(traindate, trainy, "*--", label="Truth")
        axs.plot(traindate, trainpred, "+--", label="Prediction")
        axs.grid(True)
        axs.set_title("Train Accuracy, RMSE: {0:.2f}".format(trainrmse))
        axs.set_xlabel("Date", fontsize=12)
        axs.set_ylabel("Liquid Rate", fontsize=12)
        axs.legend(loc="best")

        if axs is None:
            display(fig)

    def evalplot(self, traindate, testdate, trainy, testy, testpred, testunc=0.15, axs=None):

        if axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(12, 4))

        unc_err, rmse_unc, inrange = self._unc_errors(self, testy, testpred, testunc)

        axs.fill_between(traindate.set_index(['datetime']).index, testy + unc_err, testy - unc_err, alpha=0.2,
                         color='gray')
        axs.plot(traindate, y, "*", label="Train Truth")
        axs.plot(testdate, testy, "--", label="Test Truth")
        axs.plot(testdate, y_pred, "--", label="Prediction")
        axs.grid(True)
        axs.set_title(
            "Test Accuracy, RMSE: {0:.2f}, Predictions Within Test Range (%): {1:.2f}".format(rmse_unc, inrange))
        axs.set_xlabel("Date", fontsize=12)
        axs.set_ylabel("Liquid Rate", fontsize=12)
        axs.legend(loc="best")

        if axs is None:
            fig.tight_layout()
            display(fig)

    def _unc_errors(self, testy, testpred, testunc):
        unc_err = testy * testunc
        # Calculate RMSE away from the truths uncertainty range:
        upp = testy + unc_err
        low = testy - unc_err
        within = ((testpred >= low) & (testpred <= upp))

        err = np.stack((within, upp, low, testpred), axis=1)
        err = pd.DataFrame(err, columns=['mask', 'upp', 'low', 'testpred'])

        def f(row):
            if row['mask'] is True:
                val = 0.0
            else:
                if row['testpred'] > row['upp']:
                    val = (row['testpred'] - row['upp']) ** 2
                elif row['testpred'] < row['low']:
                    val = (row['testpred'] - row['low']) ** 2
                else:
                    val = None
            return val

        err['dif'] = err.apply(f, axis=1)
        rmse_unc = (np.sqrt(err['dif'].sum() / err.shape[0]))
        inliers = err['mask'].sum()
        inrange = inliers / err.shape[0] * 100

        return unc_err, rmse_unc, inrange

    def fitpredict(self, model, traindataset, testdataset, param_grid=None, splits=10):

        assert len(traindataset.X.columns) == len(
            testdataset.X.columns), "Train feature set is not the same length as test feature set"

        trainX = traindataset.X
        trainy = np.squeeze(traindataset.y)
        testX = testdataset.X

        if param_grid:
            kcv = KFold(n_splits=splits, shuffle=True, random_state=42)
            clf = GridSearchCV(model, param_grid, cv=kcv)
            fittedmodel = clf.fit(trainX, trainy)
        else:
            fittedmodel = model.fit(trainX, trainy)

        return fittedmodel.predict(testX)

    def fitpredict_naivemodels(self, traindataset, testdataset, errmetric="r2", eval_all=False):
        assert (errmetric == "r2") or (errmetric == "rmse"), "error metric can only be r2 or rmse"

        r2_comp = {}
        rmse_comp = {}
        model_list = {}
        eval_pred = {}

        testX = testdataset.X

        names = ["LinearRegression", "ElasticNet", "Lasso", "Ridge", "PLSRegression", "RandomForests"]
        models = [Pipeline(steps=[('ss', StandardScaler()), ('lr', LinearRegression())]),
                  Pipeline(steps=[('ss', StandardScaler()), ('lr', ElasticNet())]),
                  Pipeline(steps=[('ss', StandardScaler()), ('lr', Lasso())]),
                  Pipeline(steps=[('ss', StandardScaler()), ('lr', Ridge())]),
                  Pipeline(steps=[('ss', StandardScaler()), ('lr', PLSRegression())]),
                  Pipeline(steps=[('ss', StandardScaler()), ('rf', RandomForestRegressor())])
                  ]
        param_grids = [{'lr__fit_intercept': [True, False], 'lr__normalize': [True, False]},
                       {'lr__alpha': [0.1, 1.0, 10, 100, 1000, 1e4],
                        'lr__l1_ratio': [0.2, 0.4, 0.6, 0.8, 1.0]},
                       {'lr__alpha': [0.1, 1.0, 10, 100, 1000, 1e4]},
                       {'lr__alpha': [0.1, 1.0, 10, 100, 1000, 1e4]},
                       {'lr__n_components': [2, 3, 4]},
                       {'rf__bootstrap': [True],
                        'rf__max_depth': [70, 80, 90],
                        'rf__max_features': [2, 3],
                        'rf__min_samples_leaf': [2, 3, 4],
                        'rf__min_samples_split': [6, 7, 8],
                        'rf__n_estimators': [50, 100, 150]}
                       ]
        print("Training")
        for name, model, param_grid in zip(names, models, param_grids):
            print("Training with Model: ", name)
            r2_comp[name], rmse_comp[name] = self.validate(model, traindataset, param_grid)
            model_list[name] = model
            if eval_all: eval_pred[name] = self.fitpredict(model, traindataset, testdataset, param_grid)

        if errmetric == "r2":
            best_model = max(r2_comp, key=r2_comp.get)
            model = model_list[best_model]
        else:
            best_model = max(rmse_comp, key=rmse_comp.get)
            model = model_list[best_model]

        print("Best naive model from cross validation found to be: ", best_model)
        if not eval_all:
            eval_pred[best_model] = self.fitpredict(model_list[best_model], traindataset, testdataset, param_grid,
                                                    splits=5)

        return eval_pred, r2_comp, rmse_comp

    def plotmodels(self, traindataset, testdataset, modelpreds, testunc=0.15):

        traindate = traindataset.date
        testdate = testdataset.date
        testy = np.squeeze(testdataset.y)
        trainy = np.squeeze(traindataset.y)

        fig, axs = plt.subplots(1, 1, figsize=(12, 4))
        axs.plot(testdate, testy, "--", label="Test Truth")
        axs.plot(traindate, trainy, "*", label="Train Truth")

        for model, pred in modelpreds.items():
            pred = np.squeeze(pred)
            unc_err, rmse_unc, inrange = self._unc_errors(testy, pred, testunc)
            axs.plot(testdate, pred, "--",
                     label="{0}: RMSE {1:.2f}, {2:.2f}% within uncertainty".format(model, rmse_unc, inrange))

        axs.grid(True)
        axs.set_title("Model Comparison".format(rmse_unc, inrange))
        axs.set_xlabel("Date", fontsize=12)
        axs.set_ylabel("Liquid Rate", fontsize=12)
        #         axs.legend(loc="best")
        axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=8, borderaxespad=0.)

        axs.fill_between(testdate.set_index(['datetime']).index, testy + unc_err, testy - unc_err, alpha=0.2,
                         color='gray')

        display(fig)
        plt.close(fig)

    def get_datasets(self):
        return self.datasets

    def get_results(self):
        return self.results


class Result:
    def __init__(self, date=None, ypred=None, ytrue=None, r2=0.0, rmse=0.0):
        self.date = date
        self.ypred = ypred
        self.ytrue = ytrue
        self.r2 = r2
        self.rmse = rmse
        if ytrue is not None:
            assert len(ypred) == len(ytrue), "predictions and truths arent the same size!"
            assert ypred is not None, "ypred not given"
            self.n = len(ytrue)
            self.r2 = metrics.r2_score(self.ytrue, self.ypred)
            self.rmse = self.calc_rmse(self.ytrue, self.ypred)

    def calc_rmse(self, ytrue, ypred):
        residual = ypred - ytrue
        sse = sum(residual ** 2)
        mse = sse / self.n
        return np.sqrt(mse)

    def plot_residual(self):
        residual = self.ypred - self.ytrue
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(self.ypred, residual, ".")
        axs[0].hlines(0, 0, 2200)
        axs[0].grid(True)
        axs[1].plot(self.date, self.ypred, ".--", label="Prediction")
        axs[1].plot(self.date, self.ytrue, ".--", label="True")
        axs[1].legend(loc='best')
        axs[1].grid(True)
        display(fig)
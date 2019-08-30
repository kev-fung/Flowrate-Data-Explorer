"""
Kevin Fung
Github: kkf18
"""

import pytest
import numpy as np
from mlflowrate.classes.subclasses.explore import Result


def test_calc_rsme_with_error():
    predictions = np.array([10, 20, 30, 40])
    target = np.array([5, 15, 25, 30])
    expected_mse = 6.614378277661476
    result = Result(ytrue=target, ypred=predictions)
    actual_mse = result.calc_rmse(result.ytrue, result.ypred)
    assert expected_mse == actual_mse


def test_calc_rsme_no_error():
    predictions = np.array([10, 20, 30, 40])
    target = np.array([10, 20, 30, 40])
    expected_mse = 0
    result = Result(ytrue=target, ypred=predictions)
    actual_mse = result.calc_rmse(result.ytrue, result.ypred)
    assert expected_mse == actual_mse
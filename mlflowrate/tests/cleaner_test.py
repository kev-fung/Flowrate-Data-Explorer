"""
Kevin Fung
Github: kkf18
"""

import pytest
import pandas as pd
from mlflowrate.tests.conftest import create_spark_data_frame
from mlflowrate.classes.subclasses.clean import DataCleaner


@pytest.fixture()
def data_cleaner(base_data):
    data_cleaner = DataCleaner(base_data.dfs)
    return data_cleaner


def test_add_new_data(data_cleaner, sql_context):
    col_names = ["datetime", "tag", "value"]
    data = ["2019-01-01T12.00.00.000+0000", "tag-2", 3000]
    data_key = "new-ow"

    data_frame = pd.DataFrame(data=[data], columns=col_names)
    spark_data_frame = create_spark_data_frame(sql_context, data_frame)

    data_cleaner.add_data({data_key: spark_data_frame})

    assert data_cleaner.dfs[data_key] == spark_data_frame
    assert data_cleaner._track_org[data_key] is False


def test_add_duplicate_data(data_cleaner, data_frames):
    with pytest.raises(AssertionError, match="there is already data with the same name! .*"):
        data_cleaner.add_data(data_frames)


def test_set_organised(data_cleaner, data_frame_names):
    name = data_frame_names[0]
    data_cleaner.set_organised(name)

    assert data_cleaner._track_org[name] is True


def test_set_organised_non_existent_data(data_cleaner):
    with pytest.raises(AssertionError, match="name of data does not exist!"):
        data_cleaner.set_organised("non-existent-ow")


def test_merge_data_invalid_axis(data_cleaner, data_frame_names):
    name_one = data_frame_names[0]
    name_two = data_frame_names[1]
    with pytest.raises(AssertionError, match="axis must be either 0 or 1"):
        data_cleaner.merge_data("merged-data", name_one, name_two, axis=99)


def test_merge_data_invalid_name_one(data_cleaner, data_frame_names):
    name_two = data_frame_names[1]
    with pytest.raises(AssertionError, match="name of first data does not exist"):
        data_cleaner.merge_data("merged-data", "invalid-name-one", name_two, axis=0)


def test_merge_data_invalid_name_two(data_cleaner, data_frame_names):
    name_one = data_frame_names[0]
    with pytest.raises(AssertionError, match="name of second data does not exist"):
        data_cleaner.merge_data("merged-data", name_one, "invalid-name-two", axis=0)


def test_merge_data_bottom_append(data_cleaner, data_frame_names):
    name_one = data_frame_names[0]
    name_two = data_frame_names[1]
    merged_name = "merged-data"
    data_cleaner.merge_data(merged_name, name_one, name_two)

    assert merged_name in data_cleaner.dfs
    assert merged_name in data_cleaner.dicts
    assert data_cleaner._track_org[merged_name] is False

    assert data_cleaner.dfs['merged-data'].first()[0] == data_cleaner.dfs[name_one].first()[0]


def test_clean_data_non_existent_data(data_cleaner):
    with pytest.raises(AssertionError, match="data must be a spark dataframe stored within dfs"):
        data_cleaner.clean_data("non-existent-ow")


def test_clean_data_no_datetime_data(data_cleaner):
    name = "ow-no-datetime"
    with pytest.raises(AssertionError, match="no datetime columns can be found in dataframe"):
        data_cleaner.drop_col(name, "datetime")
        data_cleaner.clean_data(name)
    _clean_test(data_cleaner, name)


def test_clean_data_nan_all(data_cleaner):
    name = "ow-nan-all"
    data_cleaner.clean_data(name, remove_nulls=True)
    _clean_test(data_cleaner, name)


def test_clean_data_nan_column(data_cleaner):
    name = "ow-nan-col"
    data_cleaner.clean_data(name, null_col="value", remove_nulls=True)
    _clean_test(data_cleaner, name)


def test_drop_col_non_existent(data_cleaner):
    with pytest.raises(AssertionError, match="data in dictionary format not found"):
        data_cleaner.drop_col("non-existent-data", "fake-feature")


def test_drop_col(data_cleaner):
    assert len(data_cleaner.dfs['ow-drop-col'].columns) == 3
    data_cleaner.drop_col("ow-drop-col", "tag", "value")
    assert len(data_cleaner.dfs['ow-drop-col'].columns) == 1


def test_select_col_non_existent_data(data_cleaner):
    with pytest.raises(AssertionError, match="data in dictionary format not found"):
        data_cleaner.select_col("non-existent-data", "fake-feature")


def test_select_col(data_cleaner):
    assert len(data_cleaner.dfs['ow-select-col'].columns) == 3
    data_cleaner.select_col("ow-select-col", "tag")
    assert len(data_cleaner.dfs['ow-select-col'].columns) == 2


def test_edit_col_type(data_cleaner):
    name = "ow-edit-cast"
    feature_col = "value"
    previous_type = [feature[1] for feature in data_cleaner.dfs[name].dtypes if feature[0] == feature_col][0]
    assert previous_type == "bigint"
    new_type = "string"
    kwargs = {"typ": new_type}
    data_cleaner.edit_col(name, feature_col, **kwargs)
    actual_type = [feature[1] for feature in data_cleaner.dfs[name].dtypes if feature[0] == feature_col][0]
    assert actual_type == new_type


def test_edit_col_name(data_cleaner):
    name = "ow-edit-col-name"
    feature_col = data_cleaner.dfs[name].columns[2]
    new_name = "val"
    kwargs = {"newname": new_name}
    data_cleaner.edit_col(name, feature_col, **kwargs)
    renamed_col = data_cleaner.dfs[name].columns[2]
    assert renamed_col == new_name


def _clean_test(data_cleaner, name):
    data_cleaner.dfs.pop(name)
    data_cleaner.dicts.pop(name)
    data_cleaner._track_org.pop(name)

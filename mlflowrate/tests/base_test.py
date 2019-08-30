import pytest
from unittest.mock import MagicMock


def test_cache_non_existent_data(base_data):
    with pytest.raises(AssertionError, match=".* does not exist in class"):
        base_data.cache_data("non-existent-data")


def test_cache_existent_data(base_data, data_frame_names):
    name = data_frame_names[0]
    base_data.cache_data(name)
    #need to assert method call with magic mock here

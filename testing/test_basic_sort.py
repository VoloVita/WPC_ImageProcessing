# =========================================================================
#
#  Copyright Ziv Yaniv
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# =========================================================================

import pytest
import numpy as np
from basic_sort.int_sort import bubble, quick, insertion


def is_sorted(int_list):
    """
    Testing oracle which utilizes Python's built in
    sorted() function.

    Ref: https://docs.python.org/3/library/stdtypes.html#list.sort
    """
    return sorted(int_list)


@pytest.fixture
def int_test_dict():
    # fixture which creates testing data for all tests
    return {
        "reverse-list": [3, 2, 1],
        "uniform-list": [1, 1, 1],
        "already-sorted": [1, 4, 8, 34, 65, 73, 109],
        "random-ints": np.random.randint(low=-10, high=200, size=5),
        "positive-ints": [9, 6, 95, 83, 4, 2, 34],
        "negative-list": [-2, -8, -1, -64, -43, -38],
        "repeating-pos-int": [3, 3, 3, 72, 72, 43, 72, 3],
        "repeating-neg-int": [-3, 3, -5, 4, 5, 0, -3],
        "two-ints": [2, 1],
        "one-int": [1],
        "empty-list": [],
    }


def test_bubble(int_test_dict):
    for testName in int_test_dict:
        function_result = bubble(int_test_dict[testName])
        assert np.array_equal(
            function_result, is_sorted(int_test_dict[testName])
        ), f" Bubble Sort failed for {testName}"


def test_quick(int_test_dict):
    for testName in int_test_dict:
        function_result = quick(int_test_dict[testName])
        assert np.array_equal(
            function_result, is_sorted(int_test_dict[testName])
        ), f" Quick Sort failed for {testName}"


def test_insertion(int_test_dict):
    for testName in int_test_dict:
        function_result = insertion(int_test_dict[testName])
        assert np.array_equal(
            function_result, is_sorted(int_test_dict[testName])
        ), f"Insertion Sort failed for {testName}"

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

"""
This module sorts lists of integers...
"""


def bubble(int_list):
    """
    Sorts a list in ascending order using the Bubble Sort algorithm.

    Parameters:
    - arr (list): The list of comparable elements to be sorted.

    Returns:
    - list: The sorted list.

    Source of this code: https://www.programiz.com/dsa/bubble-sort
    """
    for i in range(len(int_list)):
        for n in range(0, len(int_list) - i - 1):
            if int_list[n] > int_list[n + 1]:
                temp = int_list[n]
                int_list[n] = int_list[n + 1]
                int_list[n + 1] = temp
    return int_list


def quick(int_list):
    """
    Sorts a list using the quicksort algorithm.

    Parameters:
    - arr (list): The list to be sorted.

    Returns:
    - list: The sorted list.
    """

    less = []
    equal = []
    greater = []

    if len(int_list) > 1:
        pivot = int_list[0]
        for x in int_list:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        return quick(less) + equal + quick(greater)
    else:
        return int_list


def insertion(int_list):
    """
    Sorts a list in ascending order using the Insertion Sort algorithm.

    Parameters:
    - int_list (list): The list of comparable elements to be sorted.

    Returns:
    - list: The sorted list.
    """
    n = len(int_list)

    # Traverse through 1 to n
    for i in range(1, n):
        key = int_list[i]  # index i is the current card
        j = i - 1

        while j >= 0 and key < int_list[j]:
            int_list[j + 1] = int_list[j]
            j -= 1

        int_list[j + 1] = key
    return int_list

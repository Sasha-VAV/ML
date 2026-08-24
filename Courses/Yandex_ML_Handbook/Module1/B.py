"""
source
https://new.contest.yandex.ru/contests/60376/problem?id=149944%2F2024_03_01%2Fcjahsa0C2E&tab=submissions
"""

import numpy as np


def most_frequent(nums: np.array):
    d = dict()
    for num in nums:
        d[num] = d.get(num, 0) + 1
    curr_max = -1
    answer = -1
    for k, v in d.items():
        if v > curr_max:
            answer = k
            curr_max = v
    return answer


A = np.array([1, 2, 3, 4, 5, 1, 2, 3, 1, 3, 3])
print(most_frequent(A))

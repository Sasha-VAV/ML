"""
source:
https://new.contest.yandex.ru/contests/60376/problem?id=149944%2F2024_03_01%2FMJ8hIP3a7E
"""

import numpy as np


def construct_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) < len(b):
        a, b = b, a
    if len(a) > len(b):
        temp = np.zeros_like(a)
        for i in range(len(b)):
            temp[i] = b[i]
        b = temp

    a = a.reshape((len(a), 1))
    b = b.reshape((len(b), 1))
    ans = np.concatenate((a, b), axis=1)
    return ans


A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print(construct_matrix(A, B))

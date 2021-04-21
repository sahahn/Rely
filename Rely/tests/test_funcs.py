from ..funcs import _test_split
import pandas as pd
import numpy as np

def test_test_split_groups():

    all_subjects = ['s1', 's2', 's3', 's4', 's5']
    stratify = None
    groups = pd.Series([1, 1, 2, 2, 3], index=all_subjects)

    for split in range(50):

        g1, g2 = _test_split(all_subjects, stratify,
                             groups, split_random_state=split)

        if 's1' in g1:
            assert 's2' in g1
        elif 's1' in g2:
            assert 's2' in g2

        if 's3' in g1:
            assert 's4' in g1
        elif 's3' in g2:
            assert 's4' in g2


def test_test_split_stratify():

    all_subjects = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    stratify = pd.Series([1, 1, 1, 1, 0, 0, 0, 0], index=all_subjects)
    groups = None

    for split in range(50):

        g1, g2 = _test_split(all_subjects, stratify,
                             groups, split_random_state=split)
        assert np.sum(stratify.loc[g1]) == np.sum(stratify.loc[g2])

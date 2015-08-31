import unittest
import utils_for_tests
from eights import utils

import numpy as np
import pandas as pd

class TestUtils(unittest.TestCase):
    def test_generate_matrix(self):
        M, y = utils_for_tests.generate_test_matrix(100, 5, 3, [float, str, int])
        print M
        print y

    def __sa_check(self, sa1, sa2):
        # This works even if both rows and columns are in different
        # orders in the two arrays
        frozenset_sa1_names = frozenset(sa1.dtype.names)
        frozenset_sa2_names = frozenset(sa2.dtype.names)
        self.assertEqual(frozenset_sa1_names,
                         frozenset_sa2_names)
        sa2_reordered = sa2[list(sa1.dtype.names)]
        sa1_set = {tuple(row) for row in sa1}
        sa2_set = {tuple(row) for row in sa2_reordered}
        print 'ctrl'
        print sa1_set
        print 'res'
        print sa2_set
        self.assertEqual(sa1_set, sa2_set)

    def test_join(self):
        # test basic inner join
        a1 = np.array([(0, 'Lisa', 2),
                       (1, 'Bill', 1),
                       (2, 'Fred', 2),
                       (3, 'Samantha', 2),
                       (4, 'Augustine', 1),
                       (5, 'William', 0)], dtype=[('id', int),
                                                  ('name', 'S64'),
                                                  ('dept_id', int)])
        a2 = np.array([(0, 'accts receivable'),
                       (1, 'accts payable'),
                       (2, 'shipping')], dtype=[('id', int),
                                                ('name', 'S64')])
        ctrl = pd.DataFrame(a1).merge(
                    pd.DataFrame(a2),
                    left_on='dept_id',
                    right_on='id').to_records(index=False)
        res = utils.join(a1, a2, 'inner', 'dept_id', 'id')
        self.__sa_check(ctrl, res)

        # test column naming rules
        a1 = np.array([(0, 'a', 1, 2, 3)], dtype=[('idx0', int),
                                    ('name', 'S1'),
                                    ('a1_idx1', int),
                                    ('idx2', int),
                                    ('idx3', int)])
        a2 = np.array([(0, 'b', 1, 2, 3)], dtype=[('idx0', int),
                                            ('name', 'S1'),
                                            ('a2_idx1', int),
                                            ('idx2', int),
                                            ('idx3', int)])
        pd1 = pd.DataFrame(a1)
        pd2 = pd.DataFrame(a2)
        ctrl = pd1.merge(
                pd2, 
                left_on=['idx0', 'a1_idx1', 'idx2'], 
                right_on=['idx0', 'a2_idx1', 'idx2'],
                suffixes=['_left', '_right']).to_records(index=False)
        res = utils.join(
                a1,
                a2, 
                'inner',
                left_on=['idx0', 'a1_idx1', 'idx2'], 
                right_on=['idx0', 'a2_idx1', 'idx2'],
                suffixes=['_left', '_right'])
        self.__sa_check(ctrl, res)

        # outer joins
        a1 = np.array(
            [(0, 'a1_0'),
             (1, 'a1_1'),
             (1, 'a1_2'),
             (2, 'a1_3'),
             (3, 'a1_4')], 
            dtype=[('idx', int), ('label', 'S64')])
        a2 = np.array(
            [(0, 'a2_0'),
             (1, 'a2_1'),
             (2, 'a2_2'),
             (2, 'a2_3'),
             (4, 'a2_4')], 
            dtype=[('idx', int), ('label', 'S64')])
        for how in ('inner', 'left', 'right', 'outer'):
            ctrl = pd.DataFrame(a1).merge(
                    pd.DataFrame(a2),
                    how=how,
                    left_on='idx',
                    right_on='idx').to_records(index=False)
            res = utils.join(
                    a1,
                    a2, 
                    how,
                    left_on='idx',
                    right_on='idx')
            print how.upper()
            print '-' * 80
            self.__sa_check(ctrl, res)


if __name__ == '__main__':
    unittest.main()

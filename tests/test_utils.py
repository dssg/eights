import unittest
from eights import utils
import utils_for_tests
from datetime import datetime

import numpy as np
import pandas as pd

class TestUtils(unittest.TestCase):
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
        self.assertEqual(sa1_set, sa2_set)


    def test_utf_to_ascii(self):
        s = u'\u03BBf.(\u03BBx.f(x x)) (\u05DC.f(x x))'
        ctrl = '?f.(?x.f(x x)) (?.f(x x))'
        res = utils.utf_to_ascii(s)
        self.assertTrue(isinstance(res, str))
        self.assertEqual(ctrl, res)

    def test_validate_time(self):
        trials = [('2014-12-12', True),
                  ('1/2/1999 8:23PM', True),
                  ('1988-08-15T13:43:01.123', True),
                  ('2014-14-12', False), # invalid month
                  ('2012', False), # Just a number
                  ('a', False), # dateutil interprets this as now
                 ]
        
        for (s, ctrl) in trials:
            self.assertEqual(utils.validate_time(s), ctrl)

    def test_str_to_time(self):
        trials = [('2014-12-12', datetime(2014, 12, 12)),
                  ('1/2/1999 8:23PM', datetime(1999, 1, 2, 20, 23)),
                  ('1988-08-15T13:43:01.123', 
                   datetime(1988, 8, 15, 13, 43, 1, 123000)),
                 ]

        for (s, ctrl) in trials:
            self.assertEqual(utils.str_to_time(s), ctrl)

    def test_cast_list_of_list_to_sa(self):
        L = [[None, None, None],
             ['a',  5,    None],
             ['ab', 'x',  None]]
        ctrl = np.array(
                [('', '', ''), 
                 ('a', '5', ''),
                 ('ab', 'x', '')],
                dtype=[('f0', 'S2'),
                       ('f1', 'S1'),
                       ('f2', 'S1')])
        conv = utils.cast_list_of_list_to_sa(L)
        self.assertTrue(np.array_equal(conv, ctrl))                 
        L = [[None, u'\u05dd\u05d5\u05dc\u05e9', 4.0, 7],
             [2, 'hello', np.nan, None],
             [4, None, None, 14L]]
        ctrl = np.array(
                [(-999, u'\u05dd\u05d5\u05dc\u05e9', 4.0, 7),
                 (2, u'hello', np.nan, -999L),
                 (4, u'', np.nan, 14L)],
                dtype=[('int', int), ('ucode', 'U5'), ('float', float),
                       ('long', long)])
        conv = utils.cast_list_of_list_to_sa(
                L, 
                col_names=['int', 'ucode', 'float', 'long'])
        self.assertTrue(utils_for_tests.array_equal(ctrl, conv))

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
            [(0, 'a1_0', 0),
             (1, 'a1_1', 1),
             (1, 'a1_2', 2),
             (2, 'a1_3', 3),
             (3, 'a1_4', 4)], 
            dtype=[('key', int), ('label', 'S64'), ('idx', int)])
        a2 = np.array(
            [(0, 'a2_0', 0),
             (1, 'a2_1', 1),
             (2, 'a2_2', 2),
             (2, 'a2_3', 3),
             (4, 'a2_4', 4)], 
            dtype=[('key', int), ('label', 'S64'), ('idx', int)])
        #for how in ('inner', 'left', 'right', 'outer'):
        merged_dtype = [('key', int), ('label_x', 'S64'), ('idx_x', int),
                        ('label_y', 'S64'), ('idx_y', int)]
        merge_algos = ('inner', 'left', 'right', 'outer')
        merged_data = [[(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3)],
                       [(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3),
                        (3, 'a1_4', 4, '', -999)], 
                       [(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3),
                        (4, '', -999, 'a2_4', 4)], 
                       [(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3),
                        (4, '', -999, 'a2_4', 4), 
                        (3, 'a1_4', 4, '', -999)]] 
        for how, data in zip(merge_algos, merged_data):
            res = utils.join(
                    a1,
                    a2, 
                    how,
                    left_on='key',
                    right_on='key')
            ctrl = np.array(data, dtype=merged_dtype)
            self.__sa_check(ctrl, res)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
import unittest

test_modules = ['test_investigate',
                'test_decontaminate',
                'test_generate',
                'test_utils',
                'test_communicate']

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(test_modules)
    unittest.TextTestRunner().run(suite)

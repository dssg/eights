#!/usr/bin/env python
import unittest

finished_modules = ['test_investigate',
                    'test_decontaminate']

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(finished_modules)
    unittest.TextTestRunner().run(suite)

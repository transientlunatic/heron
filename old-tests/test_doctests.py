import unittest
import doctest
import heron

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(heron))
    return tests

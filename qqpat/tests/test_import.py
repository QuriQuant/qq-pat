from unittest import TestCase

import qqpat

class TestFake(TestCase):
    def test_is_string(self):
        s = qqpat.__version__
        self.assertTrue(isinstance(s, basestring))
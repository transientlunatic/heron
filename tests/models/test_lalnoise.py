import unittest

from heron.models.lalnoise import AdvancedLIGO
from . import _GenericPSD

class TestAdvancedLIGO(_GenericPSD):

    @classmethod
    def setUpClass(self):
        self.psd_model = AdvancedLIGO

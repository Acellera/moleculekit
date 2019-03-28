# (c) 2015-2018 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import abc


class Projection(abc.ABC):
    """
    Parent class for all trajectory projecting classes. Defines abstract functions.
    """
    def __init__(self):
        self._cache = {}

    @abc.abstractmethod
    def project(self, mol):
        """ Subclasses need to implement and overload this method """
        return

    @abc.abstractmethod
    def getMapping(self, mol):
        return

    @abc.abstractmethod
    def _calculateMolProp(self, mol, props='all'):
        return

    def _setCache(self, mol):
        resdict = self._calculateMolProp(mol)
        self._cache.update(resdict)

    def _getMolProp(self, mol, prop):
        if prop in self._cache:
            resdict = self._cache
        else:
            resdict = self._calculateMolProp(mol, [prop] if prop != 'all' else 'all')
            
        if prop == 'all':
            return resdict
        else:
            return resdict[prop]

    def copy(self):
        """ Produces a deep copy of the object
        """
        from copy import deepcopy
        return deepcopy(self)

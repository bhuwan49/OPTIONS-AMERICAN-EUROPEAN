# -*- coding: utf-8 -*-

"""
optnlib.exceptions
~~~~~~~~~~~~~~~~~~

Exceptions raised by :mod:`optnlib`.
"""


class OptnlibError(Exception):
    """Base class for all optnlib errors."""


class InvalidInputError(OptnlibError, ValueError):
    """Raised when a pricing input is outside its valid domain."""
